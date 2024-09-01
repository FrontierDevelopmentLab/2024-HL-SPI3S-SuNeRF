import argparse
import os
from datetime import datetime
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as cm
from sunpy.map import Map
from sunpy.map.header_helper import get_observer_meta
from tqdm import tqdm
# For density and temperature rendering
from sunerf.rendering.density_temperature_tracing import DensityTemperatureRadiativeTransfer
from sunerf.evaluation.loader import ModelLoader
from sunerf.model.stellar_model import SimpleStar
from sunerf.model.mhd_model import MHDModel
from sunerf.model.sunerf import DensityTemperatureSuNeRFModule
import glob
import yaml
import torch
from sunerf.model.model import NeRF_DT


class ImageRender:
    """ Class that renders images from a given model output
    """
    def __init__(self, render_path):
        """

        Parameters
        ----------
        render_path : str
            Path to save rendered images

        Returns
        -------
        None
        """
        # Path to save rendered images 
        self.render_path = render_path

    def frame_to_jpeg(self, filename, observer_name, images, wavelengths, vmin=None, vmax=None, overwrite=True):
        r""" Method that saves an image from a viewpoint as the ith frame as jpg file

        Parameters
        ----------
        i : int
            frame number
        observer_name : str
            name of the observer
        images : numpy array
            model output image
        wavelengths : list

        Returns
        -------
        None
        """

        # Get the min and max values for the colormap
        if vmin is None:
            vmin = np.percentile(images, 1, axis=(0,1))
        if vmax is None:
            vmax = np.percentile(images, 99, axis=(0,1))

        for n, wavelength in enumerate(wavelengths):
            # Only save image if it doesn't exist
            output_path = f"{self.render_path}/{observer_name}/{wavelength}"
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            # Save image as jpg
            img_path = f'{output_path}/{filename}.jpg'

            # Get the image
            image = images[:, :, n]
            # Get the colormap
            cmap = plt.get_cmap(f'sdoaia{wavelength}').copy()

            # Draw the image
            fig_sizex = 4
            fig_sizey = 4
            fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout=False)
            spec = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=1.00, bottom=0.00, top=1.00, wspace=0.00) 
            ax = fig.add_subplot(spec[:, :])
            ax.imshow(image, cmap=cmap, norm='log', vmin=vmin[n], vmax=vmax[n])
            ax.set_axis_off()
            plt.draw()
            
            # Verify if files exists and whether to overwrite.
            if (not os.path.exists(img_path) or overwrite) or (os.path.exists(img_path) and overwrite):
                plt.savefig(img_path, format='jpeg', dpi=300)
            # If file exists AND overwrite is false
            else:
                print(f"File exists in image path: {img_path} and overwrite is set to False. Skipping...")
            plt.close('all')



    def frame_to_fits(self, filename, observer_name, observer_file, images, wavelengths, resolution, overwrite=False):
        r"""Method that saves an image from a viewpoint as the ith frame as fits file

        Parameters
        ----------
        i : int
            frame number
        observer_name : str
            name of the observer
        observer_file : str
            path to observer file
        images : numpy array
            model output image
        wavelengths : list
            wavelengths to render
        resolution : int
            resolution of the images
        overwrite : bool
            whether to overwrite existing files

        Returns
        -------
        None
        """
     
       
        s_map = Map(observer_file)   
        # modify header meta to get proper resolution and wavelength
        s_map = s_map.resample(resolution)
        
        for n, wavelength in enumerate(wavelengths):
        
            s_map.meta["WAVELNTH"] = wavelength 

            # overwrite with model data for each wavelength
            new_s_map = Map(images[:, :, n], s_map.meta)
            
            # Create output directory if it doesn't exist
            output_path = f"{self.render_path}/{observer_name}/{wavelength}/"
            os.makedirs(output_path, exist_ok=True)
            img_path = f'{output_path}/{filename}.fits'
            
            # Verify if files exists and whether to overwrite.
            if (not os.path.exists(img_path) or overwrite) or (os.path.exists(img_path) and overwrite):
                new_s_map.save(img_path, overwrite=True)
            # If file exists AND overwrite is false
            else:
                print(f"File exists in image path: {img_path} and overwrite is set to False. Skipping...")
            # Close maps
            new_s_map = None

        # Close maps
        s_map = None


def load_observer_meta(path_to_file):
    """ Main function to load observer data
    
    Parameters
    ----------
    path_to_file : str
        Path to AIA files

    Returns
    --------
    lat : float
        Latitude of the observer
    lon : float
        Longitude of the observer
    dist : float
        Distance of the observer from the Sun
    time : str
        Time of the observation
    """
    # Read AIA image 
    s_map = Map(path_to_file)
    # Extract observation time and satellite position when AIA produced image
    sat_coords = s_map.observer_coordinate 
    coord_meta = get_observer_meta(sat_coords)
    lat = coord_meta['hglt_obs']  # latitude [degree]
    lon = coord_meta['hgln_obs']  # longitude [degree]
    dist = coord_meta['dsun_obs']  # instrument distance in units [m]
    # Convert into expected units/coordinate system for the render
    dist = dist*u.m.to(u.au)  # conversion to [AU] with astropy
    # Extract observation time -- first condition for SDO (t_obs), second condition for STEREO (date-obs)
    time = s_map.meta['t_obs'] if ('t_obs' in s_map.meta) else s_map.meta['date-obs']

    return lat, lon, dist, time


if __name__ == '__main__':
    """ Main function to render images from a given model output
    
    Parameters
    ----------
    path_to_aia_file : str
        Path to AIA files
    render_path : str
        Path to save rendered images
    resolution : int
        Resolution of the images
    batch_size : int
        Batch size for rendering
    output_format : str
        Output format of the images
    wavelengths : list
        Wavelengths to render
        
    Returns
    -------
    Imager renders.
    """

    # Parse command line arguments
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str)
    args = p.parse_args()

    # Load configuration
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    render_path = config['render_path']
    render_format = config['render_format']
    overwrite = config['overwrite']
    observer_names = config['observer_names']
    observer_dir = config['observer_dir']
    observer_wl = config['observer_wl']
    observer_res = config['observer_res']
    observer_ref = config['observer_ref']
    batch_size = config['batch_size']
    model = config['model']
    enforce_solar_rotation = config['enforce_solar_rotation']
    
    # Find files and metadata for each observer
    observer_files = [sorted(glob.glob(f"{dir}/*.fits")) for dir in observer_dir]
    observer_meta = [[load_observer_meta(filepath) for filepath in tqdm(files)] for files in observer_files]
   
    # Reference map for module from the first file
    s_map = Map(observer_ref)
    # Extracting reference time from the map's observation time
    s_map_t = datetime.strptime(s_map.meta['t_obs'] if ('t_obs' in s_map.meta) else s_map.meta['date-obs'],
                                '%Y-%m-%dT%H:%M:%S.%f')
    
    # Initialization of the density and temperature model (Simple star analytical model or MHD model)
    if model == 'SimpleStar':
        # Dummy timesteps
        t_i = 0
        t_f = 1
        t_shift = 0
        dt = 1.0
        # Model
        model = SimpleStar
        model_config = {}
        
    elif model == 'MHDModel':
        # Path to MHD data
        data_path = '/mnt/disks/data/MHD'
        
        # List of all density files
        density_files = sorted(glob.glob(os.path.join(data_path, 'rho', '*.h5')))
        
        # Identify timesteps for first and last file
        t_i = int(density_files[0].split('00')[1].split('.h5')[0])
        t_f = int(density_files[-1].split('00')[1].split('.h5')[0])
        t_shift = 0
        # Timestep = 1 hour
        dt = 3600.0
        
        # Model
        model = MHDModel
        model_config = {'data_path': data_path}
        
    elif model == 'SuNeRF':
        
        # path to nerf data # TODO: confrim this is correct for nerf
        data_path = '/mnt/disks/data/MHD' 
        
        # List of all density files # TODO: confrim this is correct for nerf
        density_files = sorted(glob.glob(os.path.join(data_path, 'rho', '*.h5')))
        
        # path to last checkpoint
        checkpoint_path = "/mnt/disks/data/sunerfs/psi/mhd_512_checkpoint_8-13/save_state.snf"

        # load SuNeRF model
        sunerf_model = torch.load(checkpoint_path)
        
        # identify timesteps # TODO: confrim this is correct for nerf
        t_i = int(density_files[0].split('00')[1].split('.h5')[0])
        t_f = int(density_files[-1].split('00')[1].split('.h5')[0])
        t_shift = 0
        dt = 3600.0
        
        # initialize example model to overwrite rendering
     #  model = NeRF_DT
        model_config = {}
        

    else:
        raise ValueError('Model not implemented')

    # Create render path directory if it doesn't exist.
    os.makedirs(render_path, exist_ok=True)

    # Iterate over the unpacked point coordinates and render images
    # Loop over observers
    for j, files in enumerate(observer_files):
        
        if model == 'SuNeRF':
            loader = ModelLoader(rendering=sunerf_model["rendering"], model=sunerf_model["rendering"].fine_model, ref_map=s_map, serial=True)
            
        else:
            rendering = DensityTemperatureRadiativeTransfer(Rs_per_ds=1, model=model,
                                                            model_config=model_config)
            loader = ModelLoader(rendering=rendering, model=rendering.fine_model, ref_map=s_map)
            
        render = ImageRender(render_path)
        # resolution = (observer_res[j], observer_res[j])*u.pix
        resolution = (256, 256)*u.pix
        # Loop over observer files
        for i, (lat, lon, d, time) in tqdm(enumerate(observer_meta[j]), total=len(observer_meta[j])):
            # Convert time to seconds (fractional) 0 to 1 value that is expected by MHD
            t = ((datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f') -
                  s_map_t).total_seconds()+t_shift*dt)/((t_f-t_i)*dt)
            
            if enforce_solar_rotation:
            # Elapsed time - difference between time from beginning of render to current time of rendering
                elapsed_t = (datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f') -
                    s_map_t).total_seconds()
                # Elapsed rotation - current rotation of the sun based on elapsed time in images
                # 360 days in 25.38 days in 24 hours and 3600 sec 
                elapsed_rotation = 360/(25.38*24*3600)*elapsed_t
                lon = lon - elapsed_rotation

            if model == 'SuNeRF':
                wl = sunerf_model['data_config']['wavelengths']
            else:
                wl = np.array(observer_wl[j])
            # Outputs 
            outputs = loader.render_observer_image(lat*u.deg, lon*u.deg, t, wl=wl, distance=d*u.AU,
                                                   batch_size=batch_size, resolution=resolution)

            # Save as fits
            if 'fits' in render_format:
                render.frame_to_fits(time, observer_names[j], files[i], outputs['image'], observer_wl[j], resolution,
                                     overwrite=overwrite)

            # Save as jpeg
            if 'jpeg' in render_format:
                # Save stats from first image
                if j == 0 and i == 0:
                    image_min = np.percentile(outputs['image'], 1, axis=(0, 1))
                    image_max = np.percentile(outputs['image'], 99, axis=(0, 1))
                render.frame_to_jpeg(time, observer_names[j], outputs['image'], observer_wl[j], vmin=image_min,
                                     vmax=image_max)

            # Clear outputs
            outputs = None
                    
