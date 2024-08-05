import glob
import os
from datetime import datetime, timedelta
import dateutil.parser as dt
import multiprocessing

import numpy as np
import random
import pandas as pd
import torch
from astropy.io.fits import getheader
from astropy import units as u
from astropy.coordinates import SkyCoord
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import Map
from torch.utils.data import Dataset, DataLoader

from s4pi.maps.train.coordinate_transformation import pose_spherical
from s4pi.maps.train.ray_sampling import get_rays
from s4pi.irradiance.utilities.data_loader import FITSDataset
from s4pi.data.utils import loadMapStack
from s4pi.maps.train.sampling import sample_non_uniform_box as sample_stratified

class NeRFDataModule(LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        
        images, poses, rays, times, focal_lengths, wavelengths, shapes, entry_heights = get_data(hparams)
        
        # keep only one image per stack for logging
        self.min_time = np.min(np.stack(times))
        self.max_time = np.max(np.stack(times))

        self.sort_batches_by_height = hparams['sort_batches_by_height']

        first_valid_wl = [wl[0,0,:].cpu().numpy().nonzero()[0][0] for wl in wavelengths]
        self.wavelengths = [int(wl.cpu().numpy()[0,0,v]) for wl, v in zip(wavelengths, first_valid_wl)]
        self.poses = np.array([pose.numpy().squeeze() for pose in poses])
        self.times = [time.cpu().numpy()[0,0] for time in times]

        # Normalize time so that the range of values matches those of our spatial coordiantes
        if self.min_time==self.max_time:
            times = [time*0 for time in times]
        else:
            times = [time - self.min_time for time in times]
            times = [time/(self.max_time-self.min_time) for time in times]
        times = [(time - 0.5)*4. for time in times]
        
        self.images = [image.cpu().numpy()[0,:,v].reshape((int(shape.cpu().numpy()[0,0]), int(shape.cpu().numpy()[0,0]))) for image, v, shape in zip(images, first_valid_wl, shapes)]

        self.working_dir = hparams['Training']['working_directory']
        os.makedirs(self.working_dir, exist_ok=True)

        N_GPUS = torch.cuda.device_count()
        self.batch_size = int(hparams['Training']['batch_size']) * N_GPUS

        num_workers = hparams['Training']['num_workers']
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2

        # use the first image for each wavelength
        # test_idx = (np.sum(np.concatenate(self.wavelengths)>0, axis=1)==self.wavelengths[0].shape[1]).nonzero()[0]
        # test_idx = [0,8]
        test_idx = [(len(images) * 5) // 6]
        # test_idx = [len(images)-1]

        valid_rays = np.concatenate([v.squeeze() for i, v in enumerate(rays) if (i in test_idx)])
        valid_times = np.concatenate([v.squeeze() for i, v in enumerate(times) if (i in test_idx)])
        valid_images = np.concatenate([v.squeeze() for i, v in enumerate(images) if (i in test_idx)])
        valid_wavelengths = np.concatenate([v.squeeze() for i, v in enumerate(wavelengths) if (i in test_idx)])
        valid_shapes = np.concatenate([v.squeeze() for i, v in enumerate(shapes) if (i in test_idx)])

        # batching
        n_batches = int(np.ceil(valid_rays.shape[0] / self.batch_size))
        valid_rays, valid_times, valid_images, valid_wavelengths, valid_shapes  = np.array_split(valid_rays, n_batches), \
                                                                    np.array_split(valid_times, n_batches), \
                                                                    np.array_split(valid_images, n_batches), \
                                                                    np.array_split(valid_wavelengths, n_batches), \
                                                                    np.array_split(valid_shapes, n_batches)

        self.valid_rays, self.valid_times, self.valid_images, self.valid_wavelengths, self.valid_shapes = valid_rays, valid_times, valid_images, valid_wavelengths, valid_shapes
        self.test_kwargs = {'focal': np.array(focal_lengths)[test_idx],
                            'resolutions': [image.shape[0] for image in np.array(images)[test_idx]]}

        rays = np.concatenate([v.squeeze() for i, v in enumerate(rays) if not (i in test_idx)])
        times = np.concatenate([v.squeeze() for i, v in enumerate(times) if not (i in test_idx)])
        images = np.concatenate([v.squeeze() for i, v in enumerate(images) if not (i in test_idx)])
        wavelengths = np.concatenate([v.squeeze() for i, v in enumerate(wavelengths) if not (i in test_idx)])
        entry_heights = np.concatenate([v.squeeze() for i, v in enumerate(entry_heights) if not (i in test_idx)])

        # Padding batches so that they are complete
        missing_rays = rays.shape[0] % self.batch_size

        # append random rays
        if missing_rays != 0:
            random_indices = (np.random.rand(missing_rays)*rays.shape[0]).astype(int)
            rays = np.concatenate((rays, rays[random_indices]), axis=0)
            times = np.concatenate((times, times[random_indices]), axis=0)
            images = np.concatenate((images, images[random_indices]), axis=0)
            wavelengths = np.concatenate((wavelengths, wavelengths[random_indices]), axis=0)
            entry_heights = np.concatenate((entry_heights, entry_heights[random_indices]), axis=0)

        # Sort batches by height, if not, shuffle
        if hparams['sort_batches_by_height']:

            # Sort rays according to height
            r = np.argsort(entry_heights)
            # Randomize indices within vertical bands
            for i in np.linspace(0,r.shape[0], hparams['n_vertical_bands']+1):
                if i>0:
                    i1 = int(i-r.shape[0]/hparams['n_vertical_bands'])
                    i2 = int(i)
                    random.shuffle(r[i1:i2])

        else:
            # shuffle
            r = np.random.permutation(rays.shape[0])

        rays, times, images, wavelengths = rays[r], times[r], images[r], wavelengths[r]


        # batching
        n_batches = int(np.ceil(rays.shape[0] / self.batch_size))
        rays, times, images, wavelengths = np.array_split(rays, n_batches), \
                                            np.array_split(times, n_batches), \
                                            np.array_split(images, n_batches), \
                                            np.array_split(wavelengths, n_batches)

        # save to working directory --> prevent memory overflow
        batch_rays = [os.path.join(self.working_dir, f'rays_batch_{i}.npy') for i in range(len(rays))]
        batch_times = [os.path.join(self.working_dir, f'times_batch_{i}.npy') for i in range(len(times))]
        batch_images = [os.path.join(self.working_dir, f'images_batch_{i}.npy') for i in range(len(images))]
        batch_wavelengths = [os.path.join(self.working_dir, f'wavelengths_batch_{i}.npy') for i in range(len(wavelengths))]

        # save npy files
        with multiprocessing.Pool(os.cpu_count()) as p:
            p.starmap(np.save, zip(batch_rays, rays))
            p.starmap(np.save, zip(batch_times, times))
            p.starmap(np.save, zip(batch_images, images))
            p.starmap(np.save, zip(batch_wavelengths, wavelengths))
        
        # to dataset
        self.train_rays, self.train_times, self.train_images, self.train_wavelengths = batch_rays, batch_times, batch_images, batch_wavelengths

        self.valid_data = BatchesDataset(self.valid_rays, self.valid_times, self.valid_images, self.valid_wavelengths, self.valid_shapes)
        self.train_data = NumpyFilesDataset(self.train_rays, self.train_times, self.train_images, self.train_wavelengths)


    def _flatten_data(self, rays, times, images, wavelengths, shapes):
        flat_rays = np.concatenate(rays)
        flat_times = np.concatenate([np.ones(i.shape[:3], dtype=np.float32).reshape((-1, 1)) * t
                                    for t, i in zip(times, images)])
        flat_images = np.concatenate([i.reshape((-1, 1)) for i in images])
        flat_wavelengths = np.concatenate([np.ones(i.shape[:3], dtype=np.float32).reshape((-1, 1)) * wavelength
                                    for wavelength, i in zip(wavelengths, images)])

        flat_shapes = np.concatenate([np.ones(i.shape[:3], dtype=np.int16).reshape((-1, 1)) * shape
                                    for shape, i in zip(shapes, images)])
        return flat_rays, flat_times, flat_images, flat_wavelengths, flat_shapes


    def train_dataloader(self):

        shuffle = True
        if self.sort_batches_by_height:
            shuffle = False
        # handle batching manually
        return DataLoader(self.train_data, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=shuffle)

    def val_dataloader(self):
        # handle batching manually
        return DataLoader(self.valid_data, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                          shuffle=False)


class BatchesDataset(Dataset):

    def __init__(self, *batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches[0])

    def __getitem__(self, idx):
        return [b[idx] for b in self.batches]


class NumpyFilesDataset(Dataset):

    def __init__(self, *paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths[0])

    def __getitem__(self, idx):
        return [np.load(files[idx]) for files in self.paths]


def read_fits_stacks(dataset, batch_size=1, num_workers = None):
    """Predict irradiance for a given set of npy image stacks using a generator.

    Parameters
    ----------
    dataset: pytorch dataset for streaming the input data.
    batch_size: number of samples to process in parallel.
    num_workers: number of workers for data preprocessing (default cpu_count / 2).

    Returns
    -------
    stacked images
    """
    # load data
    num_workers = os.cpu_count()//2 if num_workers is None else num_workers
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    for imgs in loader:
        yield imgs


def get_data(config_data):
    debug = config_data['Debug']
    data_path = config_data['Data']['data_path']
    s_maps = sorted(glob.glob(data_path, recursive=True))

    # if config_data['stack_wavelengths']:
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
        wavelengths[wavelengths==195] = 193
        wavelengths[wavelengths==284] = 211
        data_sources[source] = {'path': path, 'wavelengths': wavelengths}
        all_wavelengths += [wavelengths]

    # Redefine wavelengths to include information about their position and presence (or lack thereof)
    all_wavelengths = np.unique(np.concatenate(all_wavelengths))
    for source in data_sources.keys():
        _, _, wl_present = np.intersect1d(data_sources[source]['wavelengths'], all_wavelengths, return_indices=True)
        wl_mask = all_wavelengths*0
        wl_mask[wl_present] = 1
        data_sources[source]['wavelengths'] = wl_mask*all_wavelengths


    # Find matches
    # Simple pairing if synthetic
    if config_data['synth']:
        for source in data_sources.keys():
            if config_data['only_ecliptic']:
                filenames = [[f for f in sorted(glob.glob(f"{data_sources[source]['path']}/{wl}/*.fits")) if np.abs(getheader(f, 0)['CRLT_OBS']) < 7] for wl in data_sources[source]['wavelengths'] if wl>0]
            else:
                filenames = [[f for f in sorted(glob.glob(f"{data_sources[source]['path']}/{wl}/*.fits"))] for wl in data_sources[source]['wavelengths'] if wl>0]
            data_sources[source]['file_stacks'] = np.array(filenames).T.tolist()
            
    # Or using date matching if using real data
    else:
        for source in data_sources.keys():
            n = 0
            for wl in data_sources[source]['wavelengths']:
                if wl>0:
                    filenames = [f for f in sorted(glob.glob(f"{data_sources[source]['path']}/{wl}/*.fits"))]
                    # account for stereo wavelengths
                    if len(filenames) == 0:
                        if wl == 193:
                            filenames = [f for f in sorted(glob.glob(f"{data_sources[source]['path']}/195/*.fits"))]
                        if wl == 211:
                            filenames = [f for f in sorted(glob.glob(f"{data_sources[source]['path']}/284/*.fits"))]
                    iso_dates = dates_from_filenames(filenames)
                    df = create_date_file_df(iso_dates, filenames, wl, debug=debug)
                    if n == 0:
                        joint_df = df
                    else:
                        joint_df = joint_df.join(df, how='inner')
                    n = n+1
            data_sources[source]['file_stacks'] = joint_df.values.tolist()

    # Load files
    rays = []
    for i, source in enumerate(data_sources.keys()):
        rays_ds = rays_from_stacksDataset(data_sources[source]['file_stacks'], data_sources[source]['wavelengths'], config_data=config_data, aia_preprocessing=False, resolution=None)
        rays += list(read_fits_stacks(rays_ds))

    images, poses, rays, times, focal_lengths, wavelengths, shapes, entry_heights = list(map(list, zip(*rays)))

    return images, poses, rays, times, focal_lengths, wavelengths, shapes, entry_heights


def dates_from_filenames(filenames):
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


def create_date_file_df(dates, files, wl, debug=False):
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


class rays_from_stacksDataset(FITSDataset):
    def __init__(self, paths, wavelengths, config_data, resolution=None, map_reproject=False, aia_preprocessing=True):
        """ Loads stacks of fits images and generates rays, focals, etc.  It returns them all flatt already

        Arguments
        ----------
            paths: list 
                list of numpy paths as string (n_samples, )
            wavelengths: np.array
                wavelengths present in the paths and the order in which they need to go in the stack

        Parameters:
            resolution: int
                resolution that the images should be converted to.  If preprocessed, this should be None
            map_reproject: bool
                apply preprocessing to remove off-limb (map to heliographic map and transform back to original view).
            aia_preprocessing: bool
                apply old aia preprocessing in the ITI loader.
            synth: bool
                Whether to read the pose from the stonyhurst (synth=True) or carrington (synth=False) keywords
        """
        super(rays_from_stacksDataset, self).__init__(paths, resolution, map_reproject, aia_preprocessing, return_meta=True)

        self.wavelengths = wavelengths
        self.config_data = config_data 

    def __getitem__(self, idx):

        stack, header = loadMapStack(self.paths[idx], resolution=self.resolution, remove_nans=True,
                                  map_reproject=self.map_reproject, aia_preprocessing=self.aia_preprocessing, return_meta=self.return_meta,
                                  normalize=False)

        stack = np.asarray(stack)

        extended_stack = np.zeros((self.wavelengths.shape[0], stack.shape[1], stack.shape[2]))
        wavelength_stack = extended_stack*0

        n = 0
        for i, wl in enumerate(self.wavelengths):
            if wl != 0:
                extended_stack[i,:,:] = stack[n,:,:]
                wavelength_stack[i,:,:] = wl
                n += 1

        s_map = Map(stack[:,:,0], header)
        # compute focal length
        scale = s_map.scale[0]  # scale of pixels [arcsec/pixel]
        if 'pc1_1' in s_map.meta.keys():
            scale = scale*s_map.meta['pc1_1']

        W = s_map.data.shape[1]  # number of pixels
        # focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to(u.deg).value * np.pi / 180)

        rsun_obs = (np.arctan(1/s_map.dsun.to(u.solRad).value)*u.rad).to(u.arcsec)/scale
        # rfov = (0.5*W/rsun_obs).value
        # print(f"shape: {s_map.data.shape[0]} - rsun_obs: {rsun_obs} - rfov: {rfov} - map scale: {s_map.scale[0]} - pc1_1: {s_map.meta['pc1_1']} - scale: {scale} - dsun")
        focal = s_map.dsun.to(u.solRad).value*rsun_obs

        time = normalize_datetime(s_map.date.datetime)
        # print(focal, s_map.dsun.to(u.solRad).value, time)

        pose = pose_spherical(-s_map.carrington_longitude.to(u.deg).value,
                            s_map.carrington_latitude.to(u.deg).value,
                            s_map.dsun.to(u.solRad).value).float().numpy()

        W = extended_stack.shape[1]  # number of pixels

        # compute focal length from Helioprojective coordinates (FOV) [pixels]
        focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to(u.deg).value * np.pi / 180)
        # focal = ((.5 * W)/s_map.meta['rsun_obs']*(s_map.meta["dsun_obs"]*u.m).to(u.solRad).value)
        focal = np.array(focal, dtype=np.float32)

        
        # get rays
        all_rays = np.stack(get_rays(extended_stack.shape[1], extended_stack.shape[2], focal, pose), -2)
        # TODO: implement subframes
        all_rays = all_rays.reshape((-1, 2, 3))

        # Getting vertical position at near distance
        rays_o, rays_d = all_rays[:, 0], all_rays[:, 1]
        # Sample query points along each ray.
        near, far = self.config_data['Sampling']['near'], self.config_data['Sampling']['far']
        query_points, z_vals = sample_stratified(torch.tensor(rays_o), 
                                                 torch.tensor(rays_d), 
                                                 near, 
                                                 far,
                                                 n_samples=10)
        # Get each point entry height
        entry_height = query_points[z_vals==torch.min(z_vals),:][:,2].numpy()

        extended_stack = extended_stack.transpose((1,2,0)).reshape((-1, self.wavelengths.shape[0]))
        wavelength_stack = wavelength_stack.transpose((1,2,0)).reshape((-1, self.wavelengths.shape[0]))
        time = time*np.ones(all_rays.shape[0])
        shape = W*np.ones(all_rays.shape[0])

        return extended_stack, pose, all_rays, time, focal, wavelength_stack, shape, entry_height        


def _load_map_data(map_path, subframe=None):
    s_map = Map(map_path)
    wavelength = s_map.wavelength.value

    # compute focal length
    scale = s_map.scale[0]  # scale of pixels [arcsec/pixel]
    if 'pc1_1' in s_map.meta.keys():
        scale = scale*s_map.meta['pc1_1']

    W = s_map.data.shape[0]  # number of pixels
    focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to(u.deg).value * np.pi / 180)



    time = normalize_datetime(s_map.date.datetime)
    # print(focal, s_map.dsun.to(u.solRad).value, time)

    pose = pose_spherical(-s_map.carrington_longitude.to(u.deg).value,
                          s_map.carrington_latitude.to(u.deg).value,
                          s_map.dsun.to(u.solRad).value).float().numpy()

    image = s_map.data.astype(np.float32)
    all_rays = np.stack(get_rays(image.shape[0], image.shape[1], focal, pose), -2)

    if subframe is not None:
        coord = SkyCoord(subframe['hgc_lon'] * u.deg, subframe['hgc_lat'] * u.deg, frame=frames.HeliographicCarrington,
                         observer=s_map.observer_coordinate)
        x, y = s_map.world_to_pixel(coord)
        x = int(x.value)
        y = int(y.value)

        w = subframe['width'] // 2
        h = subframe['height'] // 2
        image = image[max(y-h, 0):y+h, max(x-w, 0):x+w]
        all_rays = all_rays[max(y-h, 0):y+h, max(x-w, 0):x+w]

    # crop to square
    min_dim = min(image.shape[:2])
    image = image[:min_dim, :min_dim]
    all_rays = all_rays[:min_dim, :min_dim]

    # all_rays = all_rays.reshape((-1, 2, 3))

    return image, pose, all_rays, time, focal, wavelength, min_dim


def normalize_datetime(date, max_time_range=timedelta(days=30)):
    """Normalizes datetime object for ML input.

    Time starts at 2010-01-01 with max time range == 2 pi
    Parameters
    ----------
    date: input date

    Returns
    -------
    normalized date
    """
    return (date - datetime(2010, 1, 1)) / max_time_range * (2 * np.pi)


def unnormalize_datetime(norm_date: float) -> datetime:
    """Computes the actual datetime from a normalized date.

    Parameters
    ----------
    norm_date: normalized date

    Returns
    -------
    real datetime
    """
    return norm_date * timedelta(days=30) / (2 * np.pi) + datetime(2010, 1, 1)

