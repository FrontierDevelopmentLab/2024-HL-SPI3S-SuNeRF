import argparse
import os

import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader

parser = argparse.ArgumentParser('Create video of ecliptic and polar views')
parser.add_argument('--chk_path', type=str)
parser.add_argument('--video_path', type=str)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=4096)
args = parser.parse_args()

chk_path = args.chk_path
video_path = args.video_path
resolution = args.resolution
resolution = (resolution, resolution) * u.pix
batch_size = args.batch_size

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path)
cmap = cm.sdoaia193  # sdo_cmaps[loader.wavelength.to_value(u.angstrom)]
avg_time = loader.start_time + (loader.end_time - loader.start_time) / 2

n_points = 20

points_1 = zip(np.ones(n_points) * 0,
               np.linspace(0, 360, n_points),
               [avg_time] * n_points,
               np.ones(n_points))

points_2 = zip(np.linspace(0, 360, n_points),
               np.ones(n_points) * 0,
               [avg_time] * n_points,
               np.ones(n_points))

points_3 = zip(np.linspace(0, 180, n_points),
               np.linspace(0, 360, n_points),
               [avg_time] * n_points,
               np.linspace(1, 1.5, n_points), )

points_4 = zip(np.linspace(180, 270, n_points),
               np.linspace(0, 90, n_points),
               [avg_time] * n_points,
               np.linspace(1.5, 0.7, n_points), )

# combine coordinates
points = list(points_1) + list(points_2) + list(points_3) + list(points_4)

img_norm = ImageNormalize(vmin=0, vmax=0.7, stretch=AsinhStretch(0.005))

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat * u.deg, lon * u.deg, time, distance=d * u.AU, batch_size=batch_size,
                                         resolution=resolution)
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    im = axs[0, 0].imshow(outputs['image'][..., 0], cmap=cm.sdoaia171, norm=img_norm, origin='lower')
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = axs[0, 1].imshow(outputs['image'][..., 1], cmap=cm.sdoaia193, norm=img_norm, origin='lower')
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = axs[0, 2].imshow(outputs['image'][..., 2], cmap=cm.sdoaia211, norm=img_norm, origin='lower')
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = axs[1, 0].imshow(outputs['mean_T'], cmap='plasma', origin='lower', vmin=5, vmax=7)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = axs[1, 1].imshow(outputs['total_ne'], cmap='viridis', origin='lower', norm='log', vmin=1, vmax=800)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = axs[1, 2].imshow(outputs['mean_absorption'], cmap='cool', origin='lower')
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    # axs[1].imshow(outputs['height_map'], cmap='plasma', vmin=1, vmax=1.2, origin='lower')
    # axs[2].imshow(outputs['absorption_map'], cmap='viridis', origin='lower')

    axs[1, 0].set_title('Mean T [log K]')
    axs[1, 1].set_title('Total N$_e$ [cm$^{-2}$]')
    axs[1, 2].set_title('Mean Absorption')

    [ax.axis('off') for ax in axs.ravel()]
    plt.tight_layout()
    fig.savefig(os.path.join(video_path, '%03d.jpg' % i), dpi=300)
    plt.close(fig)
