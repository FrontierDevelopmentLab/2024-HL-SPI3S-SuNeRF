
import numpy as np
import torch
import random

from astropy import units as u
from torch import nn

from s4pi.maps.train.coordinate_transformation import pose_spherical
from s4pi.maps.train.model import PositionalEncoder, GaussianEncoder, BesselEncoder
from s4pi.maps.train.ray_sampling import get_rays
from s4pi.maps.train.volume_render import nerf_forward
from s4pi.maps.utilities.data_loader import normalize_datetime


class SuNeRFLoader:

    def __init__(self, state_path, path2=None, resolution=None, focal=None, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        state = torch.load(state_path)
        self.sampling_kwargs = state['sampling_kwargs']
        self.focal = focal if focal is not None else state['test_kwargs']['focal']
        self.start_time = state['start_time']
        self.end_time = state['end_time']
        self.config = state['config']

        self.encoder_kwargs = state['encoder_kwargs']

        np.random.seed(self.config['random_seed'])
        torch.manual_seed(self.config['random_seed'])
        random.seed(self.config['random_seed']) 

        print(self.encoder_kwargs)
        # if self.config['Encoders']['use_gaussian_encoder']:
        #     encoder = GaussianEncoder(**self.encoder_kwargs)

        # else:
        # self.encoder_kwargs['log_space'] = self.config['Encoders']['log_space']
        # encoder = PositionalEncoder(**self.encoder_kwargs)

        # Encoders
        if self.config['Encoders']['encoding'].lower() == 'gaussian':
            encoder = GaussianEncoder(**self.encoder_kwargs)
        elif self.config['Encoders']['encoding'].lower() == 'bessel':
            encoder = BesselEncoder(**self.encoder_kwargs)
        else:
            encoder = PositionalEncoder(**self.encoder_kwargs)


        self.encoding_fn = lambda x: encoder(x)
        # self.coarse_model = nn.DataParallel(state['coarse_model']).to(device)
        # self.fine_model = nn.DataParallel(state['fine_model']).to(device)

        if state['coarse_model'] is not None:
            self.coarse_model = state['coarse_model'].to(device)
        else:
            self.coarse_model = None
        self.fine_model = state['fine_model'].to(device)

        self.device = device

    def load_observer_image(self, lat, lon, time,
                            distance=(1 * u.AU).to(u.solRad).value,
                            strides=1, batch_size=4096):
        r"""function to render image

        Args
        -----
        lat: float
            latitude of the pose
        long: float
            longitude of the pose
        time: datetime isodate
            time of the pose  

        Params:
        -------      
        distance: float
            distance of the pose
        strides: int
            Render one in how many pixels
        batch_size:
            batch_size to be sent to memory during the rendering process

        Returns:
        --------
        outputs: dict
            dictionary with all the goodies of inference
        """


        with torch.no_grad():
            # convert to pose
            target_pose = pose_spherical(lon, lat, distance).numpy()
            # load rays
            rays_o, rays_d = get_rays(self.resolution, self.resolution, self.focal, target_pose)
            rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)
            img_shape = rays_o[::strides, ::strides].shape[:2]
            rays_o = rays_o[::strides, ::strides].reshape([-1, 3]).to(self.device)
            rays_d = rays_d[::strides, ::strides].reshape([-1, 3]).to(self.device)

            time = normalize_datetime(time)
            if self.start_time==self.end_time:
                self.times = self.times*0
            else:
                self.times = self.times - self.start_time
                self.times = self.times/(self.end_time-self.start_time)


            flat_time = (torch.ones(img_shape) * time).view((-1, 1)).to(self.device)
            # make batches
            rays_o, rays_d, time = torch.split(rays_o, batch_size), \
                                   torch.split(rays_d, batch_size), \
                                   torch.split(flat_time, batch_size)

            outputs = {'channel_map': [], 'height_map': [], 'absorption_map': []}
            for b_rays_o, b_rays_d, b_time in zip(rays_o, rays_d, time):
                b_outs = nerf_forward(b_rays_o, b_rays_d, b_time, self.coarse_model, self.fine_model,
                                      encoding_fn=self.encoding_fn,
                                      **self.sampling_kwargs)
                outputs['channel_map'] += [b_outs['channel_map'].cpu()]
                outputs['height_map'] += [b_outs['height_map'].cpu()]
                outputs['absorption_map'] += [b_outs['absorption_map'].cpu()]
            outputs = {k: torch.cat(v).view(img_shape).numpy() for k, v in
                       outputs.items()}
            return outputs

    def run_inference(self, query_points, time):
        r"""
            Function that runs inference on the fine and coarse models
        
        Inputs
        ------
        query_points: numpy array
            Array with coordinates to run inference on
        time: float
            time of inference
        """

        with torch.no_grad():
            query_points = torch.tensor(query_points)

            # add time to query points
            exp_times = torch.tensor(time).repeat(query_points.shape[0], 1)
            query_points_time = torch.cat([query_points, exp_times], -1).float()  # --> (x, y, z, t)

            # Prepare points --> encoding.
            enc_query_points = self.encoding_fn(query_points_time.view(-1, 4))

            raw_f = self.fine_model(enc_query_points.to(self.device))
            if self.coarse_model is not None:
                raw_c = self.coarse_model(enc_query_points.to(self.device))
            else:
                raw_c = None

        return raw_c, raw_f                       


