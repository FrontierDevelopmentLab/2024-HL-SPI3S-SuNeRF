import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from tqdm import tqdm
import yaml
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map, make_fitswcs_header

# Add utils module to load stacks
_root_dir = os.path.abspath(__file__).split('/')[:-4]
_root_dir = os.path.join('/',*_root_dir)
sys.path.append(_root_dir)

from s4pi.maps.evaluation.make_video import VideoRender
from s4pi.maps.models.simple_atmosphere_model import simple_star
from s4pi.maps.utilities.fibonacci_sphere import fibonacci_sphere

class simple_star_render(VideoRender):
    r"""Class to store poses, and render images of a simple model star
    """
    def __init__(self, hparams, vparams):
        super(simple_star_render, self).__init__(vparams)
        r"""
        Arguments
        ----------
        main_args : string
            path to hyperperams.yaml file with training configuration
        video_args : string
            path to hyperperams.yaml file with training configuration
        """

        self.hparams = {}
        # Read config dir
        for key in hparams.keys():
            self.hparams[key] = hparams[key]

        self.star = simple_star(hparams, self.vparams['Imager']['wavelengths'], self.device, aia_response_path=f'{_root_dir}/s4pi/maps/utilities')

        self.wavelengths = self.vparams['Imager']['wavelengths']
        self.resolution = {}
        for wavelength in self.wavelengths:
            self.resolution[wavelength] = self.hparams['Model']['resolution']/self.vparams['Imager']['strides']


    def save_frame_as_fits(self, i, point, itype='imager', strides=4, half_fov=1.3, obs_date='2014-04-01T00:00:00.000'):
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

        r, lat, lon, time = point

        # Calculate new focal length
        focal = (.5 * self.hparams['Model']['resolution']) * (1 * u.AU).to(u.solRad)/(half_fov*u.solRad)/r
        focal = np.array(focal, dtype=np.float32)

        # focal = 2*half_fov

        outputs = self.star.load_observer_image(lat, -lon, time, focal=focal, batch_size=self.vparams['Render']['batch_size'], strides=strides)
        predicted = outputs['channel_map']
        
        # Create new header
        new_observer = SkyCoord(-lon*u.deg, lat*u.deg, r*u.AU, obstime=obs_date, frame='heliographic_stonyhurst')

        out_shape = predicted.shape
        out_ref_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=new_observer.obstime,
                                frame='helioprojective', observer=new_observer,
                                rsun=696000*u.km)

        scale = 360/np.pi*np.tan(((half_fov*u.solRad*r)/(1 * u.AU).to(u.solRad)).value)/out_shape[0] * u.deg
        scale = scale.to(u.arcsec)/u.pix

        for j, wavelength in enumerate(self.vparams['Imager']['wavelengths']):

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

            output_path =  f"{self.vparams['path_to_save_video']}/{itype}/{wavelength}"
            os.makedirs(output_path, exist_ok=True)
            img_path = f'{output_path}/{str(i).zfill(3)}_w{wavelength}_lat{np.round(lat,1)}_lon{np.round(lon,1)}_r{np.round(r,2)}_T{np.round(time,2)}.fits'


            # create dummy sunpy map
            s_map = Map(predicted[:,:,j], out_header)
            s_map.save(img_path, overwrite=True)


    def assemble_panel(self, points_sp, points_car):
        r"""For single frame it plots the whole panel (views, irradiance, and orbit)
        Args
        ------
        points_sp : np.array
            r, lat,lon coordinates of viewpoints
        points_car : np.array
            x,y,z coordinates of viewpoints
        """

        # Size definitions
        dpi = 400
        pxx = 1000   # Horizontal size of each panel
        pxy = 1000   # Vertical size of each panel

        nph = 4     # Number of horizontal panels
        npv = 4     # Number of vertical panels 

        # Padding
        padv  = 50  #Vertical padding in pixels
        padv2 = 50  #Vertical padding in pixels between panels
        padh  = 50 #Horizontal padding in pixels at the edge of the figure
        padh2 = 50  #Horizontal padding in pixels between panels

        # Figure sizes in pixels
        fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
        fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

        # Conversion to relative units
        ppxx   = pxx/fszh      # Horizontal size of each panel in relative units
        ppxy   = pxy/fszv      # Vertical size of each panel in relative units
        ppadv  = padv/fszv     #Vertical padding in relative units
        ppadv2 = padv2/fszv    #Vertical padding in relative units
        ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units

        # save as gif using imageio
        #looping through the stacks of EUV images
        for i in tqdm(range(0,len(self.maps_and_stacks)), desc='Panel frames', position=1, leave=True):

            # Only do image if it doesn't exist
            output_path =  f"{self.vparams['path_to_save_video']}"
            os.makedirs(output_path, exist_ok=True)
            img_path = f'{output_path}/{str(i).zfill(3)}.jpg'

            fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi, facecolor='w')

            axes = {}
            #adding the EUV Images
            for j, wavelength in enumerate(self.vparams['Imager']['wavelengths']):

                cmap = plt.get_cmap(f'sdoaia{wavelength}').copy()
                axes[wavelength] = fig.add_axes([ppadh+ (j%2)*ppxx, ppadv-(j//2)*ppxy, ppxx, ppxy], projection=self.maps_and_stacks[i][wavelength])
                axes[wavelength].set_title("")
                self.plot_euv_image(self.maps_and_stacks[i][wavelength], axes[wavelength], str(wavelength)+'Å', cmap, vmin=None, vmax=None)
            
            fig.savefig(img_path, bbox_inches='tight', dpi = dpi, pad_inches=0)
            plt.close(fig)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='../hyperparams.yaml', required=False)
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_config_path', default = '../make_video.yaml', required=False)
    video_args = parser.parse_args()
    with open(video_args.video_config_path, 'r') as stream:
        video_config_data = yaml.load(stream, Loader=yaml.SafeLoader)    
    
    star_render_object = simple_star_render(config_data, video_config_data)

    points_car = np.array(fibonacci_sphere(300))

    lat = np.arcsin(points_car[:,2])*180/np.pi
    lon = np.arctan2(points_car[:,0], points_car[:,1])*180/np.pi
    r = lat*0+1
    t = lat*0

    points_sp = np.concatenate((r[:,None], lat[:,None], lon[:,None], t[:,None]), axis=-1)

    # points_sp = [np.array([1,0,0,0])]

    for i, point in tqdm(enumerate(points_sp), total=len(points_sp),position=0, desc='point'): 
        star_render_object.save_frame_as_fits(i, point, itype='imager', strides=star_render_object.vparams['Imager']['strides'], half_fov=star_render_object.vparams['Render']['half_fov'])


    star_render_object.load_all_imager_stacks()
    star_render_object.assemble_panel(points_sp, points_sp)

    # image_path =  f"{star_render_object.vparams['path_to_save_video']}"
    # star_render_object.images2video(image_path)