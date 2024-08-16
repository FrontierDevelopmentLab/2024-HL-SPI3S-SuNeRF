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
from itertools import repeat, chain 
import multiprocessing
from sunpy.coordinates import frames
from sunerf.data.ray_sampling import get_rays
import dateutil as dt
from sunerf.data.utils import loadMapStack
from astropy import units as u
from skimage.measure import block_reduce



class MultiThermalDataModule(BaseDataModule):

    def __init__(self, data_path, working_dir, Rs_per_ds=1, seconds_per_dt=86400, ref_time=None,
                 batch_size=int(2 ** 10), debug=False, target_resolution=None, aia_preprocessing=False, **kwargs):
        self.target_resolution = target_resolution
        self.aia_preprocessing = aia_preprocessing
        os.makedirs(working_dir, exist_ok=True)

        data_dict = self.get_data(data_path=data_path, Rs_per_ds=Rs_per_ds, debug=debug, seconds_per_dt=seconds_per_dt, ref_time=ref_time)

        # unpack data
        images = data_dict['image']
        rays = data_dict['all_rays']
        times = data_dict['time']
        wavelengths = data_dict['wavelength']
        shapes = data_dict['shape']
        instruments = data_dict['instrument']

        # log_overview(images, data_dict['pose'], np.unique(times), cmap, seconds_per_dt, ref_time)

        # select test image
        valid_index = len(images) // 6

        valid_rays, valid_times, valid_images, valid_wavelengths, valid_shapes, valid_instruments = rays[valid_index], times[valid_index], images[valid_index], wavelengths[valid_index], shapes[valid_index], instruments[valid_index]

        # load all training rays
        train_index = (np.arange(len(images)) != valid_index).nonzero()[0].tolist()
        # Index list using list of indices
        rays = np.concatenate([rays[i] for i in train_index], axis=0)
        times = np.concatenate([times[i] for i in train_index], axis=0)
        images = np.concatenate([images[i] for i in train_index], axis=0)
        wavelengths = np.concatenate([wavelengths[i] for i in train_index], axis=0)
        instruments = np.concatenate([instruments[i] for i in train_index], axis=0)

        # shuffle
        r = np.random.permutation(rays.shape[0])
        rays, times, images, wavelengths, instruments = rays[r], times[r], images[r], wavelengths[r], instruments[r]

        # save npy files
        # create file names
        logging.info('Save batches to disk')
        npy_rays = os.path.join(working_dir, 'rays_batches.npy')
        npy_times = os.path.join(working_dir, 'times_batches.npy')
        npy_images = os.path.join(working_dir, 'images_batches.npy')
        npy_wavelengths = os.path.join(working_dir, 'wavelengths_batches.npy')
        npy_instruments = os.path.join(working_dir, 'instruments_batches.npy')

        # save to disk
        np.save(npy_rays, rays)
        np.save(npy_times, times)
        np.save(npy_images, images)
        np.save(npy_wavelengths, wavelengths)
        np.save(npy_instruments, instruments)

        # adjust batch size
        N_GPUS = torch.cuda.device_count()
        batch_size = int(batch_size) * N_GPUS

        # init train dataset
        train_dataset = MmapDataset({'target_image': npy_images, 'rays': npy_rays, 'time': npy_times, 'wavelength': npy_wavelengths, 'instrument': npy_instruments},
                                    batch_size=batch_size)

        valid_dataset = ArrayDataset({'target_image': valid_images, 'rays': valid_rays, 'time': valid_times, 'wavelength': valid_wavelengths, 'instrument': valid_instruments},
                                     batch_size=batch_size)

        config = {'type': 'D_T', 'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_time': ref_time,
                  'wavelengths': valid_wavelengths[0], 'resolution': valid_shapes,
                  'times': valid_times[0], 'instrument': valid_instruments[0]}
        super().__init__({'tracing': train_dataset}, {'test_image': valid_dataset},
                         start_time=times.min(), end_time=times.max(),
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
            dates.append(dt.parser.isoparse(f'{date}T{time}'))
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

    def get_data(self, data_path, Rs_per_ds, seconds_per_dt, ref_time, debug=False):

        # Load data
        s_maps = sorted(glob.glob(data_path+'/**/*.fits', recursive=True))

        # Find unique data sources/instruments
        data_source_paths = np.unique(['/'.join(s_map.split('/')[:-2]) for s_map in s_maps])

        # Create dictionaries characterizing datasources
        data_sources = {}
        max_channels_n = 0
        for path in data_source_paths:
            source = path.split('/')[-1]
            wavelength_paths = [wv for wv in next(os.walk(path))[1]]
            wavelength_paths.sort()
            # Create wavelengths and substitute STEREO ITI conversions for the AIA equivalent
            wavelengths = np.sort(np.array([int(wl) for wl in wavelength_paths]))
            max_channels_n = np.max([max_channels_n, len(wavelengths)])
            if 'aia' in source.lower():
                instrument = 0
            elif 'euvia' in source.lower():
                instrument = 1
            elif 'euvib' in source.lower():
                instrument = 2
            elif 'eui' in source.lower():
                instrument = 3
            else:
                instrument = 0
                
            data_sources[source] = {'path': path, 'wavelengths': wavelengths, 'instrument': instrument}

        # Redefine wavelengths to include information about their position and presence (or lack thereof)
        all_wavelengths = np.zeros((max_channels_n)).astype(int)
        for source in data_sources.keys():
            wl_present = all_wavelengths.copy()
            wl_present[0:len(data_sources[source]['wavelengths'])] = np.array(data_sources[source]['wavelengths'])
            data_sources[source]['wavelengths'] = wl_present
                
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
                debug_index = np.min([2, joint_df.shape[0]])
                joint_df = joint_df.iloc[0:debug_index, :]
            data_sources[source]['file_stacks'] = joint_df.values.tolist()

        data_dict = {}
        for i, source in enumerate(data_sources.keys()):

            # TODO: Add aia_preprocessing as a parameter for when working with multi-thermal observations
            with multiprocessing.Pool(os.cpu_count()) as p:
                data = [v for v in
                        tqdm(p.imap(self._load_map_data, zip(data_sources[source]['file_stacks'], repeat(Rs_per_ds), repeat(data_sources[source]['wavelengths']),
                                                             repeat(seconds_per_dt), repeat(ref_time), repeat(source), repeat(data_sources[source]['instrument']))), 
                             total=len(data_sources[source]['file_stacks']), desc='Loading data')]

            if i==0:
                for k in data[0].keys():
                    data_dict[k] = [d[k] for d in data]
            else:
                for k in data[0].keys():
                    data_dict[k] += [d[k] for d in data]

        return data_dict


    def _load_map_data(self, data):

        stack_path, Rs_per_ds, wavelengths, seconds_per_dt, ref_time, source, instrument = data

        aia_preprocessing = False
        resolution = None
        if 'aia' in source.lower():
            aia_preprocessing = self.aia_preprocessing

        imager_stack = loadMapStack(stack_path, resolution=resolution, remove_nans=True,
                                    map_reproject=False, aia_preprocessing=aia_preprocessing, 
                                    apply_norm=False, percentile_clip=None)
        
        downscaling_factor = 1 
        if self.target_resolution is not None:
            downscaling_factor = int(imager_stack.shape[1] / self.target_resolution)
        if downscaling_factor > 1:
            imager_stack = block_reduce(imager_stack, (1,downscaling_factor,downscaling_factor), func=np.mean)
            # Read first file
            s_map = Map(stack_path[0]).resample((imager_stack.shape[1], imager_stack.shape[1]) * u.pix)
        else:
            s_map = Map(stack_path[0])
            
        time = normalize_datetime(s_map.date.datetime, seconds_per_dt, ref_time)
        pose = pose_spherical(-s_map.carrington_longitude.to(u.rad).value,
                            s_map.carrington_latitude.to(u.rad).value,
                            s_map.dsun.to_value(u.solRad) / Rs_per_ds).float().numpy()

        image = imager_stack.astype(np.float32)
        img_coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
        all_rays = np.stack(get_rays(img_coords, pose), -2)

        all_rays = all_rays.reshape((-1, 2, 3))

        extended_stack = np.zeros((wavelengths.shape[0], imager_stack.shape[1], imager_stack.shape[2])).astype(np.float32)
        wavelength_stack = extended_stack*0

        n = 0
        for i, wl in enumerate(wavelengths):
            if wl != 0:
                extended_stack[i, :, :] = image[n, :, :]
                wavelength_stack[i, :, :] = wl
                n += 1

        extended_stack = extended_stack.transpose((1,2,0)).reshape((-1, wavelengths.shape[0]))
        wavelength_stack = wavelength_stack.transpose((1,2,0)).reshape((-1, wavelengths.shape[0])).astype(int)
        time = (time*np.ones(all_rays.shape[0])[:, None]).astype(np.float32)
        shape = imager_stack.shape[1:]
        instrument = (instrument*np.ones(all_rays.shape[0])).astype(int)       

        return {'image': extended_stack, 'pose': pose, 'all_rays': all_rays, 'time': time, 'wavelength':wavelength_stack, 'shape': shape, 'instrument': instrument}
