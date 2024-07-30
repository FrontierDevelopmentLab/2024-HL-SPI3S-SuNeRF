import logging
import os
import glob
import numpy as np
import pandas as pd
import torch
from functools import partial
from sunpy.map import Map, all_coordinates_from_map
from dateutil.parser import parse
from sunerf.data.dataset import MmapDataset, ArrayDataset
from sunerf.data.date_util import normalize_datetime
from sunerf.data.loader.base_loader import BaseDataModule
from sunerf.train.callback import log_overview
from sunerf.train.coordinate_transformation import pose_spherical
from tqdm import tqdm
from itertools import repeat
import multiprocessing
from sunpy.coordinates import frames
from sunerf.data.ray_sampling import get_rays
import datetime as dt
from sunerf.data.loader.utils import loadMapStack
from astropy import units as u


class MultiThermalDataModule(BaseDataModule):

    def __init__(self, data_path, working_dir, Rs_per_ds=1, seconds_per_dt=86400, ref_time=None,
                 batch_size=int(2 ** 10), debug=False, cmap='gray', **kwargs):
        os.makedirs(working_dir, exist_ok=True)

        data_dict = self.get_data(data_path=data_path, Rs_per_ds=Rs_per_ds, debug=debug)

        o_times = data_dict['time']

        # normalize datetime
        ref_time = parse(ref_time) if ref_time is not None else min(o_times)
        times = np.array([normalize_datetime(t, seconds_per_dt, ref_time) for t in o_times], dtype=np.float32)

        # unpack data
        images = data_dict['image']
        rays = data_dict['all_rays']

        log_overview(images, data_dict['pose'], times, cmap, seconds_per_dt, ref_time)

        # select test image
        test_idx = len(images) // 6
        mask = np.ones(len(images), dtype=bool)
        mask[test_idx] = False

        valid_rays, valid_times, valid_images = rays[~mask], times[~mask], images[~mask]

        # load all training rays
        rays, times, images = rays[mask], times[mask], images[mask]

        # flatten rays
        rays = rays.reshape((-1, 2, 3))
        times = np.ones_like(images) * times[:, None, None]  # broadcast time to image shape
        times = times.reshape(-1, 1)
        images = images.reshape(-1, 1)

        # shuffle
        r = np.random.permutation(rays.shape[0])
        rays, times, images = rays[r], times[r], images[r]

        # save npy files
        # create file names
        logging.info('Save batches to disk')
        npy_rays = os.path.join(working_dir, 'rays_batches.npy')
        npy_times = os.path.join(working_dir, 'times_batches.npy')
        npy_images = os.path.join(working_dir, 'images_batches.npy')

        # save to disk
        np.save(npy_rays, rays)
        np.save(npy_times, times)
        np.save(npy_images, images)

        # adjust batch size
        N_GPUS = torch.cuda.device_count()
        batch_size = int(batch_size) * N_GPUS

        # init train dataset
        train_dataset = MmapDataset({'target_image': npy_images, 'rays': npy_rays, 'time': npy_times},
                                    batch_size=batch_size)

        valid_rays = valid_rays.reshape((-1, 2, 3))
        valid_times = np.ones_like(valid_images) * valid_times[:, None, None]
        valid_times = valid_times.reshape(-1, 1)
        valid_images = valid_images.reshape(-1, 1)

        valid_dataset = ArrayDataset({'target_image': valid_images, 'rays': valid_rays, 'time': valid_times},
                                     batch_size=batch_size)

        config = {'type': 'emission', 'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_time': ref_time,
                  'wcs': data_dict['wcs'], 'resolution': data_dict['resolution'], 'wavelength': data_dict['wavelength'],
                  'times': o_times, 'cmap': cmap}
        super().__init__({'tracing': train_dataset}, {'test_image': valid_dataset},
                         start_time=o_times.min(), end_time=o_times.max(),
                         Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt, ref_time=ref_time,
                         module_config=config, **kwargs)
        
    def dates_from_filenames(self, filenames):
        """ create dates from filenames assuming that filenames have the letter 'T' separatin date from time.

        Parameters
        ----------
        filenames: list
            list of lists each member of the list is a list including path names of the files.

        Returns
        -------
        Lists with datetimes.
        """

        dates = []
        for path in filenames:
            file = path.split('/')[-1]  
            date = file.split('T')[0][-10:]
            time = file.split('T')[1].split('_')[0].split('.')[0]
            if len(time)==2:
                time += ':00'
            dates.append(dt.isoparse(f'{date}T{time}'))
        return dates
    
    def create_date_file_df(self, dates, files, wl, debug=False):
        """ Parse a list of datetimes and files into dataframe

        Parameters
        ----------
        dates: list of datets
        files: list of filepaths

        Returns
        -------
        pandas df with datetime index and paths
        """
        df1 = pd.DataFrame(data={'dates':dates, f'files_{wl}':files}) # 171
        df1['dates'] = df1['dates'].dt.round('5min')
        # Drop duplictaes in case datetimes round to the same value
        df1 = df1.drop_duplicates(subset='dates', keep='first')
        df1 = df1.set_index('dates')

        if debug:
            df1 = df1.iloc[::debug,:]

        return df1

    def get_data(self, data_path, Rs_per_ds, debug=False):

        # Load data
        s_maps = sorted(glob.glob(data_path, recursive=True))

        # Find unique data sources/instruments
        data_source_paths = np.unique(['/'.join(s_map.split('/')[:-2]) for s_map in s_maps])

        # Create dictionaries characterizing datasources
        data_sources = {}
        all_wavelengths = []
        for path in data_source_paths:
            source = path.split('/')[-1]
            wavelength_paths = [wv.decode("utf-8") for wv in next(os.walk(path))[1]]
            wavelength_paths.sort()
            # Create wavelengths and substitute STEREO ITI conversions for the AIA equivalent
            wavelengths = np.sort(np.array([int(wl) for wl in wavelength_paths]))
            data_sources[source] = {'path': path, 'wavelengths': wavelengths}
            all_wavelengths += [wavelengths]

        # Redefine wavelengths to include information about their position and presence (or lack thereof)
        all_wavelengths = np.unique(np.concatenate(all_wavelengths))
        for source in data_sources.keys():
            _, _, wl_present = np.intersect1d(data_sources[source]['wavelengths'], all_wavelengths, return_indices=True)
            wl_mask = all_wavelengths*0
            wl_mask[wl_present] = 1
            data_sources[source]['wavelengths'] = wl_mask*all_wavelengths
                
        # Create a dataframe with the dates and filenames of the data
        for source in data_sources.keys():
            n = 0
            for wl in data_sources[source]['wavelengths']:
                if wl>0:
                    filenames = [f for f in sorted(glob.glob(f"{data_sources[source]['path']}/{wl}/*.fits"))]
                    iso_dates = self.dates_from_filenames(filenames)
                    df = self.create_date_file_df(iso_dates, filenames, wl, debug=debug)
                    if n == 0:
                        joint_df = df
                    else:
                        joint_df = joint_df.join(df, how='inner')
                    n = n+1

            if debug:
                debug_index = np.min([10, joint_df.shape[0]])
                joint_df = joint_df.iloc[0:debug_index, :]
            data_sources[source]['file_stacks'] = joint_df.values.tolist()

        data_dict = {}
        for i, source in enumerate(data_sources.keys()):

            # Extract filenames for stacks
            imager_files = []
            imager_columns = [col for col in data_sources[source]['file_stacks'].columns if 'files' in col]

            for index, row in tqdm(data_sources[source]['file_stacks'].iterrows()):
                imager_files.append(row[imager_columns].tolist())  # (channel, files)            

            # TODO: Add aia_preprocessing as a parameter for when working with multi-thermal observations
            with multiprocessing.Pool(os.cpu_count()) as p:
                data = [v for v in
                        tqdm(p.imap(self._load_map_data, zip(imager_files, repeat(Rs_per_ds))), total=len(imager_files), desc='Loading data')]

            if i==0:
                for k in data[0].keys():
                    data_dict[k] = np.stack([d[k] for d in data], axis=0)
            else:
                for k in data[0].keys():
                    data_dict[k] = np.stack([data_dict[k], np.stack([d[k] for d in data], axis=0)], axis=0)                 


        # TODO: Complete the dictionary

        # # Load files
        # rays = []
        # for i, source in enumerate(data_sources.keys()):
        #     rays_ds = rays_from_stacksDataset(data_sources[source]['file_stacks'], data_sources[source]['wavelengths'], config_data=config_data, aia_preprocessing=False, resolution=None)
        #     rays += list(read_fits_stacks(rays_ds))

        # images, poses, rays, times, focal_lengths, wavelengths, shapes, entry_heights = list(map(list, zip(*rays)))

        # return images, poses, rays, times, focal_lengths, wavelengths, shapes, entry_heights


    def get_data(self, data_path, Rs_per_ds, debug=False):
        files = sorted(glob.glob(data_path))
        if debug:
            files = files[::10]

        with multiprocessing.Pool(os.cpu_count()) as p:
            data = [v for v in
                    tqdm(p.imap(self._load_map_data, zip(files, repeat(Rs_per_ds))), total=len(files), desc='Loading data')]
        data_dict = {}
        for k in data[0].keys():
            data_dict[k] = np.stack([d[k] for d in data], axis=0)

        ref_map = Map(files[0])
        data_dict['resolution'] = ref_map.data.shape
        data_dict['wcs'] = ref_map.wcs
        data_dict['wavelength'] = ref_map.wavelength

        return data_dict


    def _load_map_data(self, data):

        stack_path, Rs_per_ds = data

        imager_stack = loadMapStack(stack_path, resolution=None, remove_nans=True,
                                    map_reproject=False, aia_preprocessing=False, 
                                    apply_norm=False, percentile_clip=None)
        
        # Read first file
        s_map = Map(imager_stack[0])
        time = s_map.date.datetime
        pose = pose_spherical(-s_map.carrington_longitude.to(u.rad).value,
                            s_map.carrington_latitude.to(u.rad).value,
                            s_map.dsun.to_value(u.solRad) / Rs_per_ds).float().numpy()

        image = imager_stack.astype(np.float32)
        img_coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
        all_rays = np.stack(get_rays(img_coords, pose), -2)

        all_rays = all_rays.reshape((-1, 2, 3))

        return {'image': image, 'pose': pose, 'all_rays': all_rays, 'time': time, }
