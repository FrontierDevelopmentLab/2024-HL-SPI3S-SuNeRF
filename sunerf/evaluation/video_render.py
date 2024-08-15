# TODO: This file needs to be better adapted once there is a model checkpoint 

import os
import glob
import torch
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Heliocentric
from sunpy.map import Map
from tqdm import tqdm
from astropy import units as u

import cv2
import spiceypy as spice
from matplotlib.lines import Line2D

from sunerf.evaluation.image_render import ImageRender
from sunerf.evaluation.loader import ModelLoader



class VideoRender(ImageRender):
    r"""Class to store poses, render images, and save video
    """
    def __init__(self, vparams):
        r"""
        Arguments
        ----------
        main_args : string
            path to hyperperams.yaml file with training configuration
        video_args : string
            path to hyperperams.yaml file with training configuration
        """

        self.vparams = {}
        # Read config dir
        for key in vparams.keys():
            self.vparams[key] = vparams[key]

        # Choose device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpus = torch.cuda.device_count()

        self.sunerf_model = torch.load(self.vparams['checkpoint_path'])
        self.reference_map = Map(self.vparams['reference_map'])
        self.loader = ModelLoader(rendering=self.sunerf_model['rendering'], model=self.sunerf_model['rendering'].fine_model, ref_map=self.reference_map, serial=True)
        self.wavelengths = self.sunerf_model['data_config']['wavelengths']

        self.resolution = (self.vparams['render']['resolution'], self.vparams['render']['resolution'])*u.pix
        self.render_targets = self.vparams['render']['render_targets']


    def draw_grid(self, ref_map, ax, lats=np.arange(90,95,30), lons=np.arange(0,365,30), **kwargs):
        r"""Method that adds a lat-lon grid to one of our map plots
        Args
        ------
        ref_map : sunpy Map
            reference map for coordinate transformations
        ax : matplotlib axes
            axes to plot on
        Params
        ------
        lats: numpy array
            latitudes in the grid
        lons: numpy array
            longitudes in the grid
        **kwargs: dictionary
            arguments to pass to the plt.plot_coord
        """

        lons_plt = np.linspace(0, 360, 360)
        for lat in lats:
            great_circle = SkyCoord(lons_plt * u.deg, np.ones(360) * lat * u.deg,
                        frame=HeliographicStonyhurst)
            # Transform to coordinate system
            hpc_great_circle = great_circle.transform_to(ref_map.coordinate_frame)
            on_disk_great_circle = hpc_great_circle.transform_to(Heliocentric).z.value < 0
            tmpLon = lons_plt.copy()
            tmpLon[on_disk_great_circle] = np.nan
            great_circle = SkyCoord(tmpLon * u.deg, np.ones(360) * lat * u.deg,
                            frame=HeliographicStonyhurst)
            hpc_great_circle = great_circle.transform_to(ref_map.coordinate_frame)

            ax.plot_coord(hpc_great_circle, **kwargs)

        lats_plt = np.linspace(-90, 90, 180)
        for lon in lons:
            great_circle = SkyCoord(np.ones(180) * lon * u.deg, lats_plt * u.deg,
                        frame=HeliographicStonyhurst)
            # Transform to coordinate system
            hpc_great_circle = great_circle.transform_to(ref_map.coordinate_frame)
            on_disk_great_circle = hpc_great_circle.transform_to(Heliocentric).z.value < 0
            tmpLat = lats_plt.copy()
            tmpLat[on_disk_great_circle] = np.nan
            great_circle = SkyCoord(np.ones(180) * lon * u.deg, tmpLat * u.deg,
                        frame=HeliographicStonyhurst)
            hpc_great_circle = great_circle.transform_to(ref_map.coordinate_frame)

            ax.plot_coord(hpc_great_circle, **kwargs)


    def initalize_kernels_and_planets(self):
        """Function to inizialize all planetary constants and orbits
        """
        os.makedirs(self.vparams['path_spice_kernel'], exist_ok=True) 
        url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp'
        os.system(f"wget -c --read-timeout=5 --no-clobber --tries=0 {url} --directory-prefix {self.vparams['path_spice_kernel']}") 

        url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc'
        os.system(f"wget -c --read-timeout=5 --no-clobber --tries=0 {url} --directory-prefix {self.vparams['path_spice_kernel']}") 

        url ='https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls'
        os.system(f"wget -c --read-timeout=5 --no-clobber --tries=0 {url} --directory-prefix {self.vparams['path_spice_kernel']}") 

        url = 'http://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/fk/rssd0002.tf'
        os.system(f"wget -c --read-timeout=5 --no-clobber --tries=0 {url} --directory-prefix {self.vparams['path_spice_kernel']}") 

        spice_kernels = glob.glob(f"{self.vparams['path_spice_kernel']}/*")
        spice.furnsh(spice_kernels)

        # Spice kernels and planets
        utc = ['Jan 1, 2020', 'Jan 1, 2021']
        etOne = spice.str2et(utc[0])

        self.sizes = np.array([2440, 6052, 6371, 3390, 69911, 58232, 25362, 24622])/6371*250
        self.orbits = [88, 225, 365, 687, 365*11.86, 365*29.46, 365*84.01, 365*164.79]
        self.colors = ['#c0bdbc', '#ffbd7c', '#286ca8', '#f27b5f', '#bfaf9b', '#f3ce88']
        self.names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn']

        self.planets = {}
        for i in np.arange(1,7):
            self.planets[i] = {}
            self.planets[i]['t'] = np.append(etOne + np.arange(0,self.orbits[i-1],0.5)*24*60*60,etOne)
            self.planets[i]['pos'], _ = spice.spkpos(str(i), self.planets[i]['t'], 'ECLIPJ2000', 'NONE', '0')
            self.planets[i]['pos']  = (self.planets[i]['pos']*u.km).to(1*u.AU).value

        posSun, _ = spice.spkpos('10', self.planets[1]['t'], 'ECLIPJ2000', 'NONE', '0')
        self.posSun = (posSun*u.km).to(1*u.AU).value     


    def draw_orbit(self, ax, point_sp, point_car):

        # Define reference point
        r, lat, lon, t = point_sp
        ref_pos = point_car[0:3]
        lim = np.sqrt(np.sum(ref_pos**2))/2
        lw = 0.5

        # find distances to define zorder
        distances = np.arange(0,7)*0
        distances[0] = np.sqrt(np.sum((self.posSun[0,:]-ref_pos)**2))
        for i in np.arange(1,7):
            distances[i] = np.sqrt(np.sum((self.planets[i]['pos'][0,:]-ref_pos)**2))

        order = distances.argsort()
        ranks = order.argsort()
        zorder = 7-ranks

        legend_dt = []	
        # Plot the Sun
        legend_size_factor = 20
        s=700/(1+np.sqrt(np.sum((self.posSun[0,:]-ref_pos)**2)))
        ax.scatter(self.posSun[0,0], self.posSun[0,1], self.posSun[0,2], c='yellow', s=s, zorder=zorder[0])
        legend_dt.append(Line2D([0], [0], color='None', lw=0, marker='o', mec='yellow', mfc='yellow', label='Sun', ms=300/legend_size_factor)) 


        for i in np.arange(1,7):
            ax.plot(self.planets[i]['pos'][:,0], self.planets[i]['pos'][:,1], self.planets[i]['pos'][:,2], c='#75756d', zorder=0, lw=lw)
            s=self.sizes[i-1]/(1+np.sqrt(np.sum((self.planets[i]['pos'][0,:]-ref_pos)**2)))
            ax.scatter(self.planets[i]['pos'][0,0], self.planets[i]['pos'][0,1], self.planets[i]['pos'][0,2], c=self.colors[i-1], s=s, zorder=zorder[i])
            if i<5:
                legend_dt.append(Line2D([0], [0], color='None', lw=0, marker='o', mec=self.colors[i-1], mfc=self.colors[i-1], label=self.names[i-1], ms=self.sizes[i-1]/legend_size_factor)) 


        elevation = lat
        azimuth = lon
        ax.view_init(elev=elevation, azim = azimuth)

        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_zlim(-lim,lim)
        ax.set_axis_off()
        ax.dis=0.0001
        ax.legend(handles=legend_dt, loc='upper right', frameon=False)
        ax.text2D(0.01, 0.99, f'r: {np.round(r,2)} AU\n$\lambda$: {np.round(lat,1)}°\n$\phi$: {np.round(lon,1)}°', horizontalalignment='left', verticalalignment='top', color = 'w', transform=ax.transAxes)	


    def camera_path(self, posOne, posTwo, n=180, r=1):
        r"""Method that estimates a path for the camera to follow between two positions in cartesian coordinates
        Args
        ------
        posOne : np.array
            starting position
        posTwo : np.array
            ending position
        Params
        ------
        n : int
            number of steps along the way
        r : float
            heigth of the path to tak
        Returns
        -------
            np.array
            points along the path iin spherical coordinates but using co-latitude
        """
        th = np.linspace(0,1,n)*np.pi

        Pone2Ptwo_c = (posOne+posTwo)/2
        Pone2Ptwo_a = posOne-Pone2Ptwo_c
        Pone2Ptwo_b = -np.cross(Pone2Ptwo_a, Pone2Ptwo_c)

        Pone2Ptwo_b = Pone2Ptwo_b/np.sqrt(np.sum(Pone2Ptwo_b**2))

        Pone2Ptwo_x = Pone2Ptwo_c[0] + np.cos(th)*Pone2Ptwo_a[0] + r*np.sin(th)*Pone2Ptwo_b[0]
        Pone2Ptwo_y = Pone2Ptwo_c[1] + np.cos(th)*Pone2Ptwo_a[1] + r*np.sin(th)*Pone2Ptwo_b[1]
        Pone2Ptwo_z = Pone2Ptwo_c[2] + np.cos(th)*Pone2Ptwo_a[2] + r*np.sin(th)*Pone2Ptwo_b[2]    

        Pone2Ptwo_r = np.sqrt(Pone2Ptwo_x**2 + Pone2Ptwo_y**2 + Pone2Ptwo_z**2)
        Pone2Ptwo_lat = np.arcsin(Pone2Ptwo_z/Pone2Ptwo_r)*180/np.pi
        Pone2Ptwo_lon = (np.arctan2(Pone2Ptwo_y, Pone2Ptwo_x)*180/np.pi)

        return np.stack((Pone2Ptwo_r, Pone2Ptwo_lat, Pone2Ptwo_lon), axis=1), np.stack((Pone2Ptwo_x, Pone2Ptwo_y, Pone2Ptwo_z), axis=1)


    def save_frames(self, points_sp, points_car):
        raise NotImplementedError("This method should be implemented in a subclass")


    def images2video(self, image_path, prefix='nerf'):
        r"""Method that converts saved images to video
        """

        for render_target in tqdm(self.render_targets, desc='Videos', position=0, leave=True):

            # Only do image if it doesn't exist
            image_path =  os.path.join(f"{self.vparams['path_to_save_video']}", render_target)        

            # save as video using opencv
            video_name = os.path.join(image_path, f'{prefix}_{render_target}_video.mp4')
            video_images = [img for img in sorted(os.listdir(image_path)) if img.endswith(".jpg")]

            frame = cv2.imread(f'{image_path}/{video_images[0]}')
            height, width, layers = frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_name, fourcc, self.vparams['render']['fps'], (width,height))

            for image in video_images:
                video.write(cv2.imread(f'{image_path}/{image}'))

            cv2.destroyAllWindows()
            video.release()

    