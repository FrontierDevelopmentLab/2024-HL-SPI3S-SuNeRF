import argparse
import os
from datetime import datetime

from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as cm
from sunpy.map import Map, make_fitswcs_header
from tqdm import tqdm
from sunerf.rendering.density_temperature import DensityTemperatureRadiativeTransfer
from sunerf.evaluation.loader import ModelLoader
from sunerf.model.stellar_model import SimpleStar

class ImageRender:
    r"""Class to store poses, render images, and save video
    """
    def __init__(self, render_path):
        r"""
        Arguments
        ----------
        main_args : string
            path to hyperperams.yaml file with training configuration
        video_args : string
            path to hyperperams.yaml file with training configuration
        """
        self.render_path = render_path

    def save_frame_as_jpg(self, i, model_output, wavelength, itype='imager'):
        r"""Method that saves an image from a viewpoint as the ith frame
        Args
        ------
        i : int
            frame number
        point : np.array
            lat,lon coordinate of viewpoint
        wavelength : int
            wavelength of image to render
        """

        # Only do image if it doesn't exist
        output_path =  f"{self.render_path}/{itype}/{wavelength}"
        os.makedirs(output_path, exist_ok=True)
        img_path = f'{output_path}/{str(i).zfill(3)}.jpg'

        if not os.path.exists(img_path):
            image = model_output/np.mean(model_output)
            cmap = plt.get_cmap(f'sdoaia{wavelength}').copy()
            plt.imsave(img_path, image, cmap=cmap, vmin=0, vmax=np.max(image))


    def save_frame_as_fits(self, i, point, model_output, wavelength, half_fov=1.3, itype='imager', obs_date='2014-04-01T00:00:00.000'):
        r"""Method that saves an image from a viewpoint as the ith frame as fits file
        Args
        ------
        i : int
            frame number
        point : np.array
            lat,lon coordinate of viewpoint
        half_fov : float
            half size field of view in solar radii 
        """

        lat, lon, d, time = point

        # save result image
        output_path =  f"{self.render_path}/{itype}/{wavelength}"
        os.makedirs(output_path, exist_ok=True)
        img_path = f'{output_path}/{str(i).zfill(3)}_w{wavelength}_lat{np.round(lat,1)}_lon{np.round(lon,1)}_r{np.round(d,2)}_T{(time.strftime("%Y%m%d-%H%M"))}.fits'

        if not os.path.exists(img_path):


            # Create new header
            new_observer = SkyCoord(-lon*u.deg, lat*u.deg, d*u.AU, obstime=obs_date, frame='heliographic_stonyhurst')

            out_shape = model_output.shape
            out_ref_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=new_observer.obstime,
                                    frame='helioprojective', observer=new_observer,
                                    rsun=696000*u.km)

            scale = 360/np.pi*np.tan(((half_fov*u.solRad*d)/(1 * u.AU).to(u.solRad)).value)/out_shape[0] * u.deg
            scale = scale.to(u.arcsec)/u.pix

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
            s_map.save(img_path, overwrite=True)


def parse_args():
    # Commands 
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--render_path', type=str)
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--output_format', type=str, default='jpg')
    p.add_argument(
        "--wavelengths",
        type=int,
        nargs="+",
        default=None,
        help="Wavelengths to render",
    )    

    args = p.parse_args()
    return args

if __name__ == '__main__':
    path_to_aia_file = "/mnt/disks/data/AIA/171/*.fits"
    s_map = Map(path_to_aia_file)

    args = parse_args()
    render_path = args.render_path
    resolution = args.resolution
    resolution = (resolution, resolution) * u.pix
    batch_size = args.batch_size
    output_format = args.output_format
    wavelengths = args.wavelengths # TODO: change to instrument specific and multi-wavelength 

    # initialization of density and temperature with simple star
    rendering = DensityTemperatureRadiativeTransfer(wavelengths = wavelengths, Rs_per_ds=1, model=SimpleStar, model_config=None) #TODO: explic. define star properties
    loader = ModelLoader(rendering=rendering, model=rendering.fine_model, ref_map=s_map)
    render = ImageRender(render_path)
    avg_time = datetime.strptime(s_map.meta['t_obs'], '%Y-%m-%dT%H:%M:%S.%f')

    os.makedirs(render_path, exist_ok=True)

    n_points = 60

    points_1 = zip(np.ones(n_points) * 0,
                np.ones(n_points) * 0,
                np.linspace(1.5, 0.2, n_points),
                [avg_time] * n_points)

    points_2 = zip(np.ones(n_points) * 0,
                np.linspace(0, 360, n_points),
                np.ones(n_points),
                [avg_time] * n_points)

    points_3 = zip(np.linspace(-90, 90, n_points),
                np.linspace(0, 360, n_points),
                np.linspace(1, 0.2, n_points),
                [avg_time] * n_points )

    # combine coordinates
    points = list(points_1) #+ list(points_2) + list(points_3)

    for i, (lat, lon, d, time) in tqdm(list(enumerate(points)), total=len(points)):
        outputs = loader.load_observer_image(lat * u.deg, lon * u.deg, time, distance=d * u.AU, batch_size=batch_size, resolution=resolution)

        for n, wavelength in enumerate(wavelengths):
            if output_format == 'jpg':
                render.save_frame_as_jpg(i, outputs['image'][:,:,n], wavelength)

            elif output_format == 'fits':
                # i, point, model_output, wavelength,
                render.save_frame_as_fits(i, (lat, lon, d, time), outputs['image'][:,:,n], wavelength, obs_date=avg_time)
            else:
                print('No valid format selected')
