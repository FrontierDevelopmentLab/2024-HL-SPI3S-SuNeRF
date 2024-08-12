import os
import shutil
from datetime import datetime

import matplotlib.axes as maxes
import numpy as np
import pandas as pd
from astropy import units as u 
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header
from tqdm import tqdm
from sunerf.model.mhd_model import MHDModel
import torch

from sunerf.rendering.density_temperature import DensityTemperatureRadiativeTransfer
from sunerf.evaluation.image_render import load_observer_meta
from sunerf.data.utils import sdo_cmaps, sdo_norms
from sunerf.evaluation.loader import ModelLoader
from sunerf.train.coordinate_transformation import spherical_to_cartesian
import matplotlib.pyplot as plt

chk_path = '/mnt/disks/data/sunerfs/psi/128/checkpoints/save_state.snf'
load_ckpt = torch.load(chk_path)
#result_path = '/mnt/disks/data/sunerfs/psi/evaluation'

sdo_map = Map('/mnt/disks/data/AIA/171/aia171_2010-05-13T00:00:07.fits')
#stereo_map = Map('/mnt/disks/data/raw/171/2012-09-06T20:00:00_A.fits')

au = (1 * u.AU).to(u.solRad).value
distance = 0.99 * au  # Distance to earth
# distance = 1.5 * au # Mars distance


# init loader
loader = ModelLoader(rendering=load_ckpt['rendering'], model=load_ckpt['rendering'].fine_model, ref_map=sdo_map)
cmap = load_ckpt['data_config']['wavelengths']

# find center point on sphere
center = spherical_to_cartesian(1, 18 * np.pi / 180, 190 * np.pi / 180)
meta_data = load_observer_meta('/mnt/disks/data/AIA/171/aia171_2010-05-13T00:00:07.fits')
lat, lon, d, time = meta_data
t = 1
#t = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')

outputs = loader.render_observer_image(lat*u.deg, lon*u.deg, t, wl=cmap, distance=distance * u.AU,
                                                   batch_size=1024, resolution=(16, 16)*u.pix)


# Plot fine image
fig, ax = plt.subplots(1, 6, figsize=(20, 10))
ax[0].imshow(outputs['fine_image'][:, :, 0], cmap='sdoaia94', origin='lower')
ax[1].imshow(outputs['fine_image'][:, :, 1], cmap='sdoaia171', origin='lower')
ax[2].imshow(outputs['fine_image'][:, :, 2], cmap='sdoaia193', origin='lower')
ax[3].imshow(outputs['fine_image'][:, :, 3], cmap='sdoaia211', origin='lower')
ax[4].imshow(outputs['fine_image'][:, :, 4], cmap='sdoaia304', origin='lower')
ax[5].imshow(outputs['fine_image'][:, :, 5], cmap='sdoaia335', origin='lower')
ax[0].set_title('94 $\AA$', fontsize=20)
ax[1].set_title('171 $\AA$', fontsize=20)
ax[2].set_title('193 $\AA$', fontsize=20)
ax[3].set_title('211 $\AA$', fontsize=20)
ax[4].set_title('304 $\AA$', fontsize=20)
ax[5].set_title('335 $\AA$', fontsize=20)
plt.savefig('MHD_128_rendered_by16x16.jpg', dpi=200, transparent=True)