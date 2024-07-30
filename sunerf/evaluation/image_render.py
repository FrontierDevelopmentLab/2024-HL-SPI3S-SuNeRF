import argparse
import os
from datetime import datetime
from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as cm
from sunpy.map import Map, make_fitswcs_header, all_coordinates_from_map, header_helper
from sunpy.map.header_helper import get_observer_meta
from tqdm import tqdm
# For density and temperature rendering
from sunerf.rendering.density_temperature import DensityTemperatureRadiativeTransfer
from sunerf.evaluation.loader import ModelLoader
from sunerf.model.stellar_model import SimpleStar
from sunerf.model.mhd_model import MHDModel
from sunpy.coordinates import get_body_heliographic_stonyhurst
import sunpy.sun.constants as constants
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

    def save_frame_as_jpg(self, i, model_output, wavelength, itype='imager', vmin=None, vmax=None):
        r""" Method that saves an image from a viewpoint as the ith frame as jpg file

        Parameters
        ----------
        i : int
            frame number
        model_output : numpy array
            model output image
        wavelength : int
            wavelength of the image

        Returns
        -------
        None
        """

        # Only save image if it doesn't exist
        output_path = f"{self.render_path}/{itype}/{wavelength}"
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Save image as jpg
        img_path = f'{output_path}/{str(i).zfill(3)}.jpg'

        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = np.nanmax(model_output)

        # Save image if it doesn't exist
        # if not os.path.exists(img_path):
        # Normalize the image
        image = model_output  # /np.nanmean(model_output)  # TODO: Consider normalizing over full time sequence
        # Get the colormap
        cmap = plt.get_cmap(f'sdoaia{wavelength}').copy()
        # Save the image
        # plt.imsave(img_path, image, cmap=cmap, vmin=0, vmax=np.nanmax(image))
        fig_sizex = 4
        fig_sizey = 4
        fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)
        spec = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=1.00, bottom=0.00, top=1.00, wspace=0.00) 
        ax = fig.add_subplot(spec[:, :])
        ax.imshow(image, cmap=cmap, norm='log', vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        plt.draw()
        plt.savefig(img_path, format='jpeg', dpi=300)


    def save_frame_as_fits(self, i, point, model_output, wavelength, half_fov=1.3,
                           itype='imager', obs_date='2014-04-01T00:00:00.000'):
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

        # Unpack or separate the point coordinates
        lat, lon, d, time = point

        # Create output directory if it doesn't exist
        output_path = f"{self.render_path}/{itype}/{wavelength}"
        os.makedirs(output_path, exist_ok=True)
        # Save image as fits
        img_path = f'{output_path}/{str(i).zfill(3)}_w{wavelength}_lat{np.round(lat,1)}_lon{np.round(lon,1)}_r{np.round(d,2)}_T{(time.strftime("%Y%m%d-%H%M"))}.fits'

        # Save image if it doesn't exist
        if not os.path.exists(img_path):

            # Create new header
            new_observer = SkyCoord(-lon*u.deg, lat*u.deg, d*u.AU, obstime=obs_date, frame='heliographic_stonyhurst')
            # Get the shape of the output image
            out_shape = model_output.shape
            # Create reference coordinate
            out_ref_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=new_observer.obstime, frame='helioprojective',
                                     observer=new_observer, rsun=696000*u.km)
            # Calculate scaling factor
            scale = 360/np.pi*np.tan(((half_fov*u.solRad*d)/(1 * u.AU).to(u.solRad)).value)/out_shape[0] * u.deg
            # Convert scale to arcsec per pixel
            scale = scale.to(u.arcsec)/u.pix

            # Create new header
            out_header = make_fitswcs_header(
                out_shape,
                out_ref_coord,
                scale=u.Quantity([scale, scale]),
                rotation_matrix=np.array([[1,0],[0,1]]),
                instrument='SPI3S',
                wavelength=wavelength*u.Angstrom
            )

            # Add sun radii information
            out_header['r_sun'] = out_shape[0]/2/half_fov

            # create dummy sunpy map
            s_map = Map(model_output, out_header)
            # Save the image
            s_map.save(img_path, overwrite=True)


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
    p.add_argument('--model', type=str, default='SimpleStar' )
    p.add_argument(
        "--wavelengths",
        type=int,
        nargs="+",
        default=None,
        help="Wavelengths to render",
    )    

    args = p.parse_args()
    return args


def load_observer_meta(path_to_file):
    '''Main function to load observer data
    
    Parameters
    ----------
    path_to_aia_file : str
        Path to AIA files

    Returns:

    '''    
    # Read AIA image 
    s_map = Map(path_to_file)
    
    # Extract observation time and satellite position when AIA produced image
    sat_coords = s_map.observer_coordinate 
    coord_meta = get_observer_meta(sat_coords)
    lat = coord_meta['hglt_obs'] #latitude [degree]
    lon = coord_meta['hgln_obs'] #longiture [degree]
    dist = coord_meta['dsun_obs'] # instrument distance in units [m]
    
    # Convert into expected units/coordinate system for the render
    dist = dist*u.m.to(u.au) # convertion to [AU] with astropy
    
    # Extract observation time 
    time = str(s_map.date)  # s_map.meta['t_obs']  # TODO: Keyword does not exist for ITI
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
    sdo_files = sorted(glob.glob("/mnt/disks/data/raw/sdo_2012_08/1h_171/*.fits"))[0:10]
    sdo_meta = [load_observer_meta(filepath) for filepath in tqdm(sdo_files)]
    stereo_a_files = sorted(glob.glob("/mnt/disks/data/raw/stereo_2012_08_converted_fov/171/*_A.fits"))[0:10]
    stereo_a_meta = [load_observer_meta(filepath) for filepath in tqdm(stereo_a_files)]
    stereo_b_files = sorted(glob.glob("/mnt/disks/data/raw/stereo_2012_08_converted_fov/171/*_B.fits"))[0:10]
    stereo_b_meta = [load_observer_meta(filepath) for filepath in tqdm(stereo_b_files)]
    s_map = Map(stereo_a_files[0])
    # print(s_map.meta)
    print(s_map.date, s_map.meta['date-obs'])
    exit()
    # Combine all observer meta data
    observer_meta = sdo_meta  + stereo_a_meta + stereo_b_meta

    # Reference map for module
    s_map = Map(sdo_files[0])
    # s_map_t = datetime.strptime(s_map.meta['t_obs'], '%Y-%m-%dT%H:%M:%S.%f')
    s_map_t = datetime.strptime(str(s_map.date), '%Y-%m-%dT%H:%M:%S.%f')

    # Initialization of the density and temperature model (Simple star analytical model or MHD model)
    if model=='SimpleStar':
        rendering = DensityTemperatureRadiativeTransfer(wavelengths=wavelengths, Rs_per_ds=1, model=SimpleStar, model_config=None)
        # Dummy timesteps
        t_i = 0
        t_f = 1
        t_shift = 0
        dt = 1.0
    elif model=='MHDModel':
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
        rendering = DensityTemperatureRadiativeTransfer(wavelengths=wavelengths, Rs_per_ds=1, model=MHDModel, model_config={'data_path': data_path})
    
    # Compute pixel intensity for a given model
    loader = ModelLoader(rendering=rendering, model=rendering.fine_model, ref_map=s_map)
    render = ImageRender(render_path)

    # Render images
    images = []
    # Iterate over the unpacked point coordinates
    for i, (lat, lon, d, time) in tqdm(enumerate(observer_meta), total=len(observer_meta)):
        # Convert time to seconds
        t = ((datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f') - s_map_t).total_seconds()+t_shift*dt)/((t_f-t_i)*dt)
        print(t * (t_f-t_i)*dt, t_i, t_f, t)
        # Outputs
        outputs = loader.load_observer_image(lat*u.deg, lon*u.deg, t, distance=d*u.AU, batch_size=batch_size, resolution=resolution)
        images.append(outputs['image'])

    # TODO: Modify for fits files
    # Create render path directory if it doesnt exist. 
    os.makedirs(render_path, exist_ok=True)
    # Iterate over wavelengths and save images
    for n, wavelength in enumerate(wavelengths):    
        for i, image in enumerate(images):
            if output_format == 'jpg':
                # Save as jpg
                if i == 0:
                    image_min = np.percentile(image, 1) 
                    image_max =  np.percentile(image, 99) 
                render.save_frame_as_jpg(i, image[:,:,n], wavelength, vmin=image_min, vmax=image_max)

            if output_format == 'fits':
                # i, point, model_output, wavelength,
                # Save as FITS
                # TODO: Pass file header information
                render.save_frame_as_fits(i, (lat, lon, d, time), image[:,:,n], wavelength)

