from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from astropy import units as u
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch import nn
import concurrent.futures
# For date normalization utilities
from sunerf.data.date_util import normalize_datetime, unnormalize_datetime
from sunerf.data.ray_sampling import get_rays
from sunerf.train.coordinate_transformation import pose_spherical

class SuNeRFLoader:

    def __init__(self, state_path, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"Loading model to device {device}")
        self.device = device

        state = torch.load(state_path)
        data_config = state['data_config']
        # Store data configuration
        self.config = data_config
        # Store wavelength
        self.wavelength = data_config['wavelength']
        self.times = data_config['times']
        # Store World Coordinate System info
        self.wcs = data_config['wcs']
        self.resolution = data_config['resolution']

        # get rendering model
        rendering = state['rendering']
        # Parallelize and move to device
        self.rendering = nn.DataParallel(rendering).to(device)
        model = rendering.fine_model
        self.model = nn.DataParallel(model).to(device)

        # Get seconds per delta time
        self.seconds_per_dt = state['seconds_per_dt']
        # Get solar radii per data scale
        self.Rs_per_ds = state['Rs_per_ds']
        # Convert to megameters per data scale
        self.Mm_per_ds = self.Rs_per_ds * (1 * u.R_sun).to_value(u.Mm)
        # Get reference time
        self.ref_time = state['ref_time']

        # Reference map
        ref_map = Map(np.zeros(self.resolution), self.wcs)
        self.ref_map = ref_map

    @property
    def start_time(self):
        return np.min(self.times)

    @property
    def end_time(self):
        return np.max(self.times)

    # Disable gradient calculation
    @torch.no_grad()
    def render_observer_image(self, lat: u, lon: u, time: datetime,
                            distance = (1 * u.AU).to(u.solRad),
                            center: Tuple[float, float, float] = None, resolution=None,
                            batch_size: int = 128):
                            #Original batch_size : 4096
                            
        # convert to pose
        target_pose = pose_spherical(-lon.to_value(u.rad), lat.to_vaslue(u.rad), distance.to_value(u.solRad), center).numpy()
        # load rays
        if resolution is not None:
            # Resample map to desired resolution
            ref_map = self.ref_map.resample(resolution)
            img_coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)
        else:
            img_coords = all_coordinates_from_map(self.ref_map).transform_to(frames.Helioprojective)

        # get rays from coordinates and pose
        rays_o, rays_d = get_rays(img_coords, target_pose)
        rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)

        img_shape = rays_o.shape[:2]
        flat_rays_o = rays_o.reshape([-1, 3]).to(self.device)
        flat_rays_d = rays_d.reshape([-1, 3]).to(self.device)

        time = normalize_datetime(time, self.seconds_per_dt, self.ref_time)
        flat_time = torch.ones_like(flat_rays_o[:, 0:1]) * time
        # make batches
        rays_o, rays_d, time = torch.split(flat_rays_o, batch_size), \
            torch.split(flat_rays_d, batch_size), \
            torch.split(flat_time, batch_size)

        outputs = {}
        # iterate over batches
        # b = batches
        for b_rays_o, b_rays_d, b_time in zip(rays_o, rays_d, time):
            b_outs = self.rendering(b_rays_o, b_rays_d, b_time)
            # iterate over outputs 
            for k, v in b_outs.items():
                if k not in outputs:
                    # initialise list if key is not in outputs
                    outputs[k] = []
                outputs[k].append(v)
        # concatenate and reshape outputs
        results = {k: torch.cat(v).view(*img_shape, *v[0].shape[1:]).cpu().numpy() for k, v in outputs.items()}
        return results

    # normalizing datetime
    def normalize_datetime(self, time):
        return normalize_datetime(time, self.seconds_per_dt, self.ref_time)

    # whats the purpose of unnormalising datetime ?
    def unnormalize_datetime(self, time):
        return unnormalize_datetime(time, self.seconds_per_dt, self.ref_time)

    # Disabling gradient calculation
    @torch.no_grad()
    def load_coords(self, query_points_npy, batch_size=2048):
        target_shape = query_points_npy.shape[:-1]
        query_points = torch.from_numpy(query_points_npy).float()

        flat_query_points = query_points.reshape(-1, 4)
        n_batches = np.ceil(len(flat_query_points) / batch_size).astype(int)

        out_list = []
        for j in range(n_batches):
            batch = flat_query_points[j * batch_size:(j + 1) * batch_size].to(self.device)
            out = self.model(batch)
            out_list.append(out.detach().cpu())

        output = torch.cat(out_list, 0).view(*target_shape, -1).numpy()
        return output


class ModelLoader(SuNeRFLoader):    
    def __init__(self, rendering, model, ref_map, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        
        self.device = device
        self.ref_map = ref_map
        self.rendering = nn.DataParallel(rendering).to(device)
        self.model = nn.DataParallel(model).to(device)   
        self.seconds_per_dt = 1
        self.ref_time = datetime.strptime(ref_map.meta['t_obs'] 
                                          if ('t_obs' in ref_map.meta) else ref_map.meta['date-obs'], 
                                          '%Y-%m-%dT%H:%M:%S.%f')

    def process_batch(self, b_rays_o, b_rays_d, b_time):
        b_outs = self.rendering(b_rays_o, b_rays_d, b_time)
        return b_outs

    def process_batch_with_index(self, index, b_rays_o, b_rays_d, b_time):
            result = self.process_batch(b_rays_o, b_rays_d, b_time)
            return index, result


    @torch.no_grad()
    def render_observer_image(self, lat: u, lon: u, time: float,
                            distance=(1 * u.AU).to(u.solRad),
                            center: Tuple[float, float, float] = None, resolution=None,
                            batch_size: int = 4096):
        """ Render observer image at a given time and location.

        Parameters
        ----------
        lat : u
            Latitude of the observer.
        lon : u
            Longitude of the observer.
        time : float
            Time of the observation.
        distance : u
            Distance of the observer from the Sun.
        center : Tuple[float, float, float], optional
            Center of the observer.
        resolution : None, optional
            Resolution of the image to render.
        batch_size : int, optional
            Batch size for rendering.
        """

        # Convert coordinates to pose
        target_pose = pose_spherical(-lon.to_value(u.rad), lat.to_value(u.rad),
                                     distance.to_value(u.solRad), center).numpy()

        # Get coordinates of the image pixels
        if resolution is not None:
            # Resample map to desired resolution
            ref_map = self.ref_map.resample(resolution)
            # Get new coordinates of the image pixels
            img_coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)
        else:
            # Get coordinates of the image pixels
            img_coords = all_coordinates_from_map(self.ref_map).transform_to(frames.Helioprojective)

        # Get rays from coordinates and pose
        # rays_o: origin of the rays
        # rays_d: direction of the rays
        rays_o, rays_d = get_rays(img_coords, target_pose)
        # Convert rays to tensors
        rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)

        # Get image shape
        img_shape = rays_o.shape[:2]
        # Flatten rays
        flat_rays_o = rays_o.reshape([-1, 3]).to(self.device)
        flat_rays_d = rays_d.reshape([-1, 3]).to(self.device)

        #self.ref_time = datetime.strptime(ref_map.meta['t_obs'], '%Y-%m-%dT%H:%M:%S.%f')
        # time = normalize_datetime(datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f'), self.seconds_per_dt, self.ref_time)
        # Create tensor of time values
        flat_time = torch.ones_like(flat_rays_o[:, 0:1]) * time
        # make batches
        rays_o, rays_d, time = torch.split(flat_rays_o, batch_size), \
            torch.split(flat_rays_d, batch_size), \
            torch.split(flat_time, batch_size)

        # Initialize outputs
        outputs = {}
        # Iterate over batches of rays and time. Use ThreadPoolExecutor for parallel processing
        
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
           futures = [executor.submit(self.process_batch, b_rays_o, b_rays_d, b_time) for b_rays_o, b_rays_d, b_time in
                      zip(rays_o, rays_d, time)]
           for future in concurrent.futures.as_completed(futures):
               b_outs = future.result()
               for k, v in b_outs.items():
                   outputs.setdefault(k, []).append(v)
                   
        results = {k: torch.cat(v).view(*img_shape, *v[0].shape[1:]).cpu().numpy() for k, v in outputs.items()}
        return results


        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self.process_batch_with_index, idx, b_rays_o, b_rays_d, b_time) for idx, (b_rays_o, b_rays_d, b_time) in 
        #                enumerate(zip(rays_o, rays_d, time))]
            
        #     for future in concurrent.futures.as_completed(futures):
        #         index, b_outs = future.result()
        #         # b_outs = future.result()
        #         for k, v in b_outs.items():
        #             outputs.setdefault(index, {}).setdefault(k, []).append(v)
        #             # outputs.setdefault(k, []).append(v)
                    
        # print(type(outputs))
        # # Reorder outputs if needed
        
        # ordered_outputs = {idx: outputs[idx] for idx in sorted(tuple(outputs))}
        # print(type(ordered_outputs))
        # tensor_tuple = tuple(ordered_outputs.values())
        # print(type(tensor_tuple))



        # results = {k: torch.cat(v).view(*img_shape, *v[0].shape[1:]).cpu().numpy() for k, v in ordered_outputs.items()}
        # return results
        
        

        # Concatenate and reshape outputs
        # k: key, v: value
        #results = {k: torch.cat(v).view(*img_shape, *v[0].shape[1:]).cpu().numpy() for k, v in outputs.items()}
       
        # results = {k: torch.cat(tuple(v)).view(*img_shape, *v[0].shape[1:]).cpu().numpy() for k, v in outputs.items()}
        # return results
       
       
       
        # results = {}
        # for k, tensor_dict in ordered_outputs.items():
        #     if isinstance(tensor_dict, dict):
        #     #Extract tensors from dictionary values
        #        tensors = list(tensor_dict.values())
        #     else:
        #     #Directly use tensors if v is not a dictionary
        #        tensors = tensor_dict
        #     if not isinstance(tensors, (list, tuple)):
        #         raise TypeError(f"Expected list or tuple of tensors for key {k}, got {type(tensors)}")
        #     if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        #         raise TypeError(f"All elements for key {k} must be tensors")
        #     # Concatenate tensors
        #     concatenated_tensor = torch.cat(tensors)
    
        #     # Reshape and convert to NumPy
        #     reshaped_tensor = concatenated_tensor.view(*img_shape, *tensors[0].shape[1:])
        #     results[k] = reshaped_tensor.cpu().numpy()
        
        # return results
