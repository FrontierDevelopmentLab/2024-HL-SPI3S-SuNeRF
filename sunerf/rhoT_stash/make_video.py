# TODO: This file needs to be better adapted once there is a model checkpoint 

import os, sys
import glob
import argparse
import multiprocessing
import yaml
import torch
import numpy as np
from itertools import repeat
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from sunpy.coordinates import HeliographicStonyhurst, Heliocentric
from sunpy.map import Map, make_fitswcs_header
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
import cv2
import imageio.v2 as imageio
from torch import nn
import spiceypy as spice
from matplotlib.lines import Line2D


# Add utils module to load stacks
_root_dir = os.path.abspath(__file__).split('/')[:-4]
_root_dir = os.path.join('/',*_root_dir)
sys.path.append(_root_dir)

from s4pi.data.utils import loadMap
from s4pi.maps.utilities.data_loader import loadAIAMap
from s4pi.irradiance.utilities.data_loader import FITSDataset
from s4pi.irradiance.inference import ipredict, ipredict_uncertainty
from s4pi.maps.evaluation.loader import SuNeRFLoader

class VideoRender:
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

        # Download irradiance checkpoints
        os.makedirs(self.vparams['Irradiance']['checkpoints_local_path'], exist_ok=True)
        os.system(f"gsutil cp {self.vparams['Irradiance']['checkpoints_remote_path']}/* {self.vparams['Irradiance']['checkpoints_local_path']}")

        self.irradiance_model_path = f"{self.vparams['Irradiance']['checkpoints_local_path']}/{self.vparams['Irradiance']['model']}"
        self.eve_normalization = np.load(f"{self.vparams['Irradiance']['checkpoints_local_path']}/{self.vparams['Irradiance']['normalization']}")	
        self.eve_wl_names = np.load(self.vparams['Irradiance']['eve_wl_names'], allow_pickle=True)        


    def read_models(self):
        r"""Mehtod to read the models to use in inference
        """        

        os.makedirs(self.vparams['Sunerf']['checkpoints_local_path'], exist_ok=True)
        os.system(f"gsutil cp {self.vparams['Sunerf']['checkpoints_remote_path']}/*_large* {self.vparams['Sunerf']['checkpoints_local_path']}")

        self.models = {}
        self.resolution = {}
        wavelengths = list(set(self.vparams['Imager']['wavelengths'] + self.vparams['Irradiance']['wavelengths']))

        for wavelength in wavelengths:

            model_file = f"{self.vparams['Sunerf']['checkpoints_local_path']}/{wavelength}_large.snf"

            if os.path.exists(model_file):

                self.models[wavelength] = SuNeRFLoader(model_file, self.device)
                self.resolution[wavelength] = self.models[wavelength].height


    def read_images_define_observer(self, distance# TODO: This file needs to be better adapted once there is a model checkpoint 
=1):
        r"""Method that defines the observer and read images
        Params
        ------
        distance : float
            distance from the detector to the center of the Sun in astronomical units
        """

        wavelengths = list(set(self.vparams['Imager']['wavelengths'] + self.vparams['Irradiance']['wavelengths']))

        if not False:

            # Normalization of the images (0 to 1)
            self.norms = {171: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=True),
                    193: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=True),
                    211: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=True),
                    304: ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.001), clip=True),}            

            self.images = {}
            self.ref_map = {}
            for wavelength in tqdm(self.wavelengths, desc='wavelength'):

                sdo_paths = sorted(glob.glob(f'{self.SDO_files_path}/{wavelength}/*.fits'))[0:2]

                # # Load maps
                with multiprocessing.Pool(os.cpu_count()) as p:
                    sdo_maps = p.starmap(loadAIAMap, zip(sdo_paths, repeat(256)))  # TODO Starmap works here.
                #     iti_a_maps = p.starmap(loadITIMap, zip(iti_a_paths, repeat(resolution))) # TODO Problem with loadITIMap + starmap
                #     iti_b_maps =  p.starmap(loadITIMap, zip(iti_b_paths, repeat(resolution))) 

                # Extract the viewpoint (-lon: change observer to counter the Sun's rotation)
                views = [((s_map.carrington_latitude.to(u.deg).value, -s_map.carrington_longitude.to(u.deg).value), s_map) for s_map in sdo_maps]

                # create times for maps
                times = [s_map.date.datetime for (lat,lon),s_map in views]
                start_time, end_time = min(times), max(times)
                times = np.array([(t - start_time) / (end_time - start_time) for t in times], dtype=np.float32)

                # images = [np.rot90(sdo_norms[wavelength](s_map.data)) for (lat, lon), s_map in views]
                images = [self.norms[wavelength](s_map.data) for (lat, lon), s_map in views]
                images = np.array(images).astype(np.float32)
                images = np.nan_to_num(images, nan = 0)
                images = images[..., None] # (N_Images, H, W, 1)
                # print(np.max(images))
                # images /= np.max(images)

                self.images[wavelength] = images
                self.images[wavelength] = np.array(images)
                self.ref_map[wavelength] = sdo_maps[0]                     
        
        else:

            self.norms = {171: ImageNormalize(vmin=0, vmax=22348.267578125, stretch=AsinhStretch(0.005), clip=True),
                            193: ImageNormalize(vmin=0, vmax=50000, stretch=AsinhStretch(0.005), clip=True),
                            211: ImageNormalize(vmin=0, vmax=13503.1240234375, stretch=AsinhStretch(0.005), clip=True),}

            self.images = {}
            for wavelength in tqdm(wavelengths):

                synth_paths = sorted(glob.glob(f'{self.synth_files_path}/AIA_{wavelength}/*'))
                with multiprocessing.Pool(os.cpu_count()) as p:
                    synth_maps = p.starmap(loadMap, zip(synth_paths, repeat(self.resolution)))

                views = [((s_map.carrington_latitude.to(u.deg).value, -s_map.carrington_longitude.to(u.deg).value), s_map) for s_map in [*synth_maps]]
                # Normalization of the images (0 to 1) for **PSI** data

                images = [self.norms[wavelength](s_map.data) for (lat, lon), s_map in views]
                times = [s_map.date.datetime for (lat,lon),s_map in views]
                start_time, end_time = min(times), max(times)
                times = np.array([0 for t in times], dtype=np.float32)
            
                images = np.array(images).astype(np.float32)
                images = np.nan_to_num(images, nan = 0)
                images = images[..., None] # (N_Images, H, W, 1)
                # images /= np.max(images)
                self.images[wavelength] = images
                self.ref_map[wavelength] = synth_maps[0]

        ref_map = views[0][1]
        scale = ref_map.scale[0] # scale of pixels [arcsec/pixel]
        W = ref_map.data.shape[0] # number of pixels

        # detector
        self.distance =  (distance * u.AU).to(u.solRad).value  #distance between camera and solar center

        # compute focal length from Helioprojective coordinates (FOV) [pixels]
        focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to(u.deg).value * np.pi/180) 
        self.focal =  torch.from_numpy(np.array(focal, dtype = np.float32))

        self.near, self.far = (1 * u.AU - 1.2 * u.solRad).to(u.solRad).value , (1 * u.AU + 1.2 * u.solRad).to(u.solRad).value


    def save_frame_as_jpg(self, i, point, wavelength, itype='imager', strides=4, half_fov=1.3):
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
        output_path =  f"{self.vparams['path_to_save_video']}/{itype}/{wavelength}"
        os.makedirs(output_path, exist_ok=True)
        img_path = f'{output_path}/{str(i).zfill(3)}.jpg'

        if not os.path.exists(img_path):

            cmap = plt.get_cmap(f'sdoaia{wavelength}').copy()
            r, lon, lat, time = point

            # Calculate new focal length
            focal = (.5 * self.models[wavelength].height) * (r * u.AU).to(u.solRad)/(half_fov*u.solRad) 
            focal = np.array(focal, dtype=np.float32)        

            outputs = self.models[wavelength].load_observer_image(lat, -lon, time, focal=focal, batch_size=self.vparams['Render']['batch_size'], strides=strides,
                                        r_decay=self.vparams['Render']['r_decay'], tau_decay=self.vparams['Render']['tau_decay'], r_cut=self.vparams['Render']['r_cut'])

            plt.imsave(img_path, outputs['channel_map'], cmap=cmap, vmin=0, vmax=1)



    def save_frame_as_fits(self, i, point, wavelength, itype='imager', strides=4, half_fov=1.3, obs_date='2014-04-01T00:00:00.000'):
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

        # save result image
        output_path =  f"{self.vparams['path_to_save_video']}/{itype}/{wavelength}"
        os.makedirs(output_path, exist_ok=True)
        img_path = f'{output_path}/{str(i).zfill(3)}_w{wavelength}_lat{np.round(lat,1)}_lon{np.round(lon,1)}_r{np.round(r,2)}_T{np.round(time,2)}.fits'

        if not os.path.exists(img_path):

            # Calculate new focal length
            focal = (.5 * self.models[wavelength].height) * (1 * u.AU).to(u.solRad)/(half_fov*u.solRad)/r
            focal = np.array(focal, dtype=np.float32)

            outputs = self.models[wavelength].load_observer_image(-lat, -lon, time, focal=focal, batch_size=self.vparams['Render']['batch_size'], strides=strides,
                                        r_decay=self.vparams['Render']['r_decay'], tau_decay=self.vparams['Render']['tau_decay'], r_cut=self.vparams['Render']['r_cut'])
            # Undo normalization
            predicted = outputs['channel_map']
            # predicted = self.norms[wavelength].inverse(outputs['channel_map'])

            # Create new header
            new_observer = SkyCoord(-lon*u.deg, lat*u.deg, r*u.AU, obstime=obs_date, frame='heliographic_stonyhurst')

            out_shape = predicted.shape
            out_ref_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=new_observer.obstime,
                                    frame='helioprojective', observer=new_observer,
                                    rsun=696000*u.km)

            scale = 360/np.pi*np.tan(((half_fov*u.solRad*r)/(1 * u.AU).to(u.solRad)).value)/out_shape[0] * u.deg
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
            s_map = Map(predicted, out_header)
            s_map.save(img_path, overwrite=True)


    def loadITIstack(self, paths, resolution):
        r"""Method that loads an ITI stack of files
        Args
        ------
        paths : list
            list of paths to load
        resolution : int
            resolution used by loadMap
        Returns
        -------
            image_stack: np.array
                stack of images
        """

        image_stack = loadMap(paths[0],resolution).data[:,:,None]
        if len(paths)>1:
            for path in paths:
                image_stack = np.append(image_stack, loadMap(path, resolution).data[:,:,None], axis=2)

        return image_stack


    def load_all_imager_stacks(self):
        r"""Method that loads all stacks and maps for plotting
        """
        paths= {}
        for wavelength in self.vparams['Imager']['wavelengths']:
            output_path =  f"{self.vparams['path_to_save_video']}/imager/{wavelength}"
            paths[wavelength] = sorted(glob.glob(output_path+'/*.fits'))

        self.maps_and_stacks = {}
        for i in tqdm(range(0,len(paths[self.vparams['Imager']['wavelengths'][0]])), desc='Panel frames', position=1, leave=True):

            stack_paths = []
            for wavelength in self.vparams['Imager']['wavelengths']:
                stack_paths.append(paths[wavelength][i])

            self.maps_and_stacks[i] = {}
            resolution = self.resolution[wavelength]/self.vparams['Imager']['strides']			
            self.maps_and_stacks[i]['stack'] = self.loadITIstack(stack_paths, resolution)
            for j, wavelength in enumerate(self.vparams['Imager']['wavelengths']):
                self.maps_and_stacks[i][wavelength] = Map(stack_paths[j]) 


    def plot_euv_image(self, s_map, ax, wavelength, cmap, vmin=0, vmax=1, flipud=False):
        r"""Method that plots a EUV image given one of our maps
        Args
        ------
        s_map : sunpy Map
            map to plot
        ax : matplotlib axes
            axes to plot on
        wavelength: string
            wavelength string to add to the top-left corner
        cmap: matplotlib colormap
            colormap for plot
        vmin: float
            minimum color scale value
        vmax: float
            maximum color scale value
        """

        x, y = np.meshgrid(*[np.arange(v.value) for v in s_map.dimensions]) * u.pixel
        if flipud:
            ax.pcolormesh(x.value, y.value, np.flipud(s_map.data), cmap = cmap, vmin = vmin, vmax = vmax)
        else:
            ax.pcolormesh(x.value, y.value, s_map.data, cmap = cmap, vmin = vmin, vmax = vmax)
        print(f'{wavelength}: {np.nanmax(np.sqrt(s_map.data))}')
        ax.text(0.01, 0.99, wavelength, horizontalalignment='left', verticalalignment='top', color = 'w', transform=ax.transAxes)	
        ax.set_axis_off()
        # self.draw_grid(s_map, ax, lats=np.arange(-90,95,30), lons=np.arange(0,365,30), lw=0.3, ls='-', c='w')


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


    def draw_irradiance(self, axs, irradiance, sun_distance=1):

        axs.bar(np.arange(0, irradiance.shape[0]), irradiance[:,0], yerr=irradiance[:,1],
            width=0.9, alpha=0.5, ecolor='red', capsize=8, color='#1288FF')
        axs.set_xticks(np.arange(0,irradiance.shape[0]))
        axs.set_xticklabels(self.eve_wl_names,rotation = 45)
        axs.set_yscale('log', nonpositive='clip')
        axs.set_ylim(1e-6, 1e-3)
        axs.yaxis.set_label_position("right")
        axs.yaxis.tick_right()
        axs.set_ylabel('Relative Irradiance log(w/m$^2$)')
        axs.set_title("Spectral Solar Irradiance")       


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


    def run_irradiance_inference(self):
        r"""method to run irradiance inference on loaded stacks
        """

        view_files = [sorted(glob.glob(f"{self.vparams['path_to_save_video']}/irradiance/{wavelength}/*.fits")) for wavelength in self.vparams['Irradiance']['wavelengths']]
        view_files = np.array(view_files).transpose()
        view_ds = FITSDataset(view_files, aia_preprocessing=False, resolution=256)

        irr_generator = ipredict_uncertainty(self.irradiance_model_path, view_ds, self.eve_normalization)

        irr_mean = []
        irr_std = []
        for i, (mean, std, _) in tqdm(enumerate(irr_generator)):
            irr_mean.append(mean)
            irr_std.append(std)

        self.total_irradiance = np.concatenate((np.array(irr_mean)[:,:,None], np.array(irr_std)[:,:,None]), axis=-1)	
        

    def images2video(self, image_path, prefix='nerf'):
        r"""Method that converts saved images to video
        """

        # save as gif using imageio
        video_name = f"{self.vparams['path_to_save_video']}/{prefix}_video.gif"
        video_images = [img for img in sorted(os.listdir(image_path)) if img.endswith(".jpg")]
        images = []
        for filename in video_images:
            images.append(imageio.imread(f'{image_path}/{filename}'))
        imageio.mimsave(video_name, images)


        # save as video using opencv
        video_name = f"{self.vparams['path_to_save_video']}/{prefix}_video.mp4"
        frame = cv2.imread(f'{image_path}/{filename}')
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, self.vparams['Render']['fps'], (width,height))

        for image in video_images:
            video.write(cv2.imread(f'{image_path}/{filename}'))

        cv2.destroyAllWindows()
        video.release()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default = '../make_video.yaml', required=False)
    video_args = parser.parse_args()
    with open(video_args.config_path, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)    
    
    video_render_object = VideoRender(config_data)
    video_render_object.read_models()

    # load viewpoints
    points_lat = np.stack(np.mgrid[0:1:, -90:1:90], -1).astype(np.float32)  # lon, lat
    points_lon = np.stack(np.mgrid[10:361:350, 0:1], -1).astype(np.float32)  # lon, lat
    # combine coordinates
    points_lat = np.moveaxis(points_lat, 0, 1).reshape([-1, 2])
    points_lon = np.moveaxis(points_lon, 0, 1).reshape([-1, 2])

    points = np.concatenate([points_lat, points_lon])
    times = np.linspace(0,1,points.shape[0])

    for wavelength in video_render_object.vparams['Imager']['wavelengths']:
        itype = 'imager' 
        image_path =  f"{video_render_object.vparams['path_to_save_video']}/{itype}/{wavelength}"
        # save all frames as jpgs
        for i, ((lon, lat), time) in tqdm(enumerate(zip(points, times)), total=len(points)): 
            point = [1, lon, lat, time]
            video_render_object.save_frame_as_jpg(i, point, wavelength, itype=itype, strides=video_render_object.vparams['Imager']['strides'], half_fov=video_render_object.vparams['Render']['half_fov'])
            if i%24 == 0 & i>0:
                video_render_object.images2video(image_path)
        video_render_object.images2video(image_path)
    