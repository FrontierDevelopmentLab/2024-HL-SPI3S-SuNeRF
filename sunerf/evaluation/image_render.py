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
from sunerf.rendering.density_temperature import DensityTemperatureRadiativeTransfer
from sunerf.evaluation.loader import ModelLoader
from sunerf.model.stellar_model import SimpleStar
from sunerf.model.mhd_model import MHDModel
import glob


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

    def frame_to_jpeg(self, i, observer_name, images, wavelengths, vmin=None, vmax=None, overwrite=True):
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
            img_path = f'{output_path}/{str(i).zfill(3)}.jpg'

            # Get the image
            image = images[:, :, n]
            # Get the colormap
            cmap = plt.get_cmap(f'sdoaia{wavelength}').copy()

            # Draw the image
            fig_sizex = 4
            fig_sizey = 4
            fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)
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


    # Modifying or changing the headers
    def frame_to_fits(self, i, observer_name, observer_file, images, wavelengths, resolution, overwrite=False):
        r"""Method that saves an image from a viewpoint as the ith frame as fits file

        Parameters
        ----------
        i : int
            frame number
        point : tuple
            observer coordinates
        model_output : numpy array
            model output image
        wavelength : int
            wavelength of the image
        headers : 
            optional headers for the fits files
        half_fov : float
            half field of view of the image
        itype : str
            type of image
        obs_date : str
            observation date

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
            
            file_name = f'{observer_name}_{i}'
            # [render_path]/[instrument]/[wl]/[fits file] 
            img_path = f'{output_path}{file_name}.fits'
            
            # Verify if files exists and whether to overwrite.
            if (not os.path.exists(img_path) or overwrite) or (os.path.exists(img_path) and overwrite):
                new_s_map.save(img_path, overwrite=True)
            # If file exists AND overwrite is false
            else:
                print(f"File exists in image path: {img_path} and overwrite is set to False. Skipping...")


def parse_args():
    """ Function to parse command line arguments

    Parameters
    ----------
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
    args : argparse.Namespace
        Parsed command line arguments
    """

    # Commands 
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--render_path', type=str)
    p.add_argument('--resolution', type=int, default=1024) 
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--output_format', type=str, default='jpg')
    p.add_argument('--model', type=str, default='SimpleStar')
    p.add_argument(
        "--wavelengths",
        type=int,
        nargs="+",
        default=None,
        help="Wavelengths to render",
    )    
    return p.parse_args()

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
    args = parse_args()
    render_path = args.render_path
    resolution = args.resolution
    resolution = (resolution, resolution) * u.pix
    batch_size = args.batch_size
    output_format = args.output_format
    wavelengths = args.wavelengths  
    model = args.model

    # TODO: Make this scenario-dependent
    # TODO: Extract full header and update it?
    # List of all fits files for SDO, STEREO-A, STEREO-B    
    sdo_files = sorted(glob.glob("/mnt/disks/data/raw/loadertest/aia/193/*.fits"))[0:1]
    sdo_meta = [load_observer_meta(filepath) for filepath in tqdm(sdo_files)]
    stereo_a_files = sorted(glob.glob("/mnt/disks/data/raw/loadertest/euvia/193/*_A.fits"))[0:1]
    stereo_a_meta = [load_observer_meta(filepath) for filepath in tqdm(stereo_a_files)]
    stereo_b_files = sorted(glob.glob("/mnt/disks/data/raw/loadertest/euvib/193/*_B.fits"))[0:1]
    stereo_b_meta = [load_observer_meta(filepath) for filepath in tqdm(stereo_b_files)]
    s_map = Map(stereo_a_files[0])

    sdo_wavelengths = [94, 171, 193, 211, 304, 335]
    stereo_a_wavelengths = [94, 171, 193, 211, 304, 335]
    stereo_b_wavelengths = [94, 171, 193, 211, 304, 335]
    
    # Combine all observer meta data into one single list
    observer_names = ["AIA", "EUVIA", "EUVIB"]
    observer_meta = [sdo_meta, stereo_a_meta, stereo_b_meta]
    observer_files = [sdo_files, stereo_a_files, stereo_b_files]
    observer_wavelengths = [sdo_wavelengths, stereo_a_wavelengths, stereo_b_wavelengths] # sdo, stereoa, stereo-b wavelengths 
   
    # Reference map for module from the first SDO-AIA file
    s_map = Map(sdo_files[0])
    
    # Extracting reference time from map's observation time
    s_map_t = datetime.strptime(s_map.meta['t_obs'] if ('t_obs' in s_map.meta) else s_map.meta['date-obs'],
                                '%Y-%m-%dT%H:%M:%S.%f')
    
    # Initialization of the density and temperature model (Simple star analytical model or MHD model)
    #TODO: adjust renderer for wavelength of each observer (modify wavelengths=observer_wavelengths[0])
    if model == 'SimpleStar':
        rendering = DensityTemperatureRadiativeTransfer(wavelengths=observer_wavelengths[0], Rs_per_ds=1, model=SimpleStar,
                                                        model_config=None)
        # Dummy timesteps
        t_i = 0
        t_f = 1
        t_shift = 0
        dt = 1.0
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
        # Define MHD model and rendering
        rendering = DensityTemperatureRadiativeTransfer(wavelengths=observer_wavelengths[0], Rs_per_ds=1, model=MHDModel,
                                                        model_config={'data_path': data_path})
    else:
        raise ValueError('Model not implemented')
    
    # Compute pixel intensity for a given model
    loader = ModelLoader(rendering=rendering, model=rendering.fine_model, ref_map=s_map)
    render = ImageRender(render_path)


    # Render images
    images = []
    # Iterate over the unpacked point coordinates
    # Create render path directory if it doesnt exist. 
    os.makedirs(render_path, exist_ok=True)
    
    for j, files in enumerate(observer_files):
        for i, (lat, lon, d, time) in tqdm(enumerate(observer_meta[j]), total=len(observer_meta[j])):
            # Convert time to seconds (fractional) 0 to 1 value that is expected by MHD
            t = ((datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f') - s_map_t).total_seconds()+t_shift*dt)/((t_f-t_i)*dt)
            # t * (t_f-t_i) + t_i --> Time step of the observation
            # Outputs
            outputs = loader.render_observer_image(lat*u.deg, lon*u.deg, t, distance=d*u.AU, batch_size=batch_size, resolution=resolution)
            image = outputs['image']
            # images.append(outputs['image'])
    
            if output_format == 'fits':
                render.frame_to_fits(i, observer_names[j], files[i], image, observer_wavelengths[j], resolution, overwrite=True)
    
            if output_format == 'jpg':
            # Save as jpg
                if j == 0 and i == 0:
                    image_min = np.percentile(image, 1, axis=(0, 1))
                    image_max = np.percentile(image, 99, axis=(0,1))
                render.frame_to_jpeg(i, observer_names[j], image, observer_wavelengths[j], vmin=image_min, vmax=image_max)
                    
