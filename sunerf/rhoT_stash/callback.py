from typing import Optional

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
import sunpy

from s4pi.data.utils import sdo_img_norm
from astropy import units as u

from s4pi.data.utils import sdo_cmaps
from s4pi.maps.utilities.data_loader import unnormalize_datetime

def plot_samples(channel_map, channel_map_coarse, height_map,
                 testimg,
                 z_vals_stratified,
                 z_vals_hierach,
                 wavelengths,
                 distance
                 ):
    # Log example images on wandb
    # # Plot example outputs

    # Remove missing wavelengths
    channel_map = channel_map[:,:,wavelengths>0]
    channel_map_coarse = channel_map_coarse[:,:,wavelengths>0]
    testimg = testimg[:,:,wavelengths>0]

    wavelengths = wavelengths[wavelengths>0]

    n_channels = wavelengths.shape[0]

    fig = plt.figure(figsize=2*np.array([n_channels+1, 3]), dpi=500)
    gs0 = fig.add_gridspec(3, n_channels+1, wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    for i in np.arange(0,n_channels):
        cmap = plt.get_cmap(f'sdoaia{int(wavelengths[i])}').copy()
        ax = fig.add_subplot(gs0[0, i])
        # ax.imshow(testimg[..., i], cmap=cmap, norm=sdo_img_norm)
        ax.imshow(testimg[..., i], cmap=cmap, norm=sdo_img_norm)
        if i==0:
            ax.set_ylabel(f'Target')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs0[1, i])
        ax.imshow(channel_map[..., i], cmap=cmap, norm=sdo_img_norm)
        if i==0:
            ax.set_ylabel(f'Prediction')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs0[2, i])
        ax.imshow(channel_map_coarse[..., i], cmap=cmap, norm=sdo_img_norm)
        if i==0:
            ax.set_ylabel(f'Coarse')
        ax.set_xticks([])
        ax.set_yticks([])


    ax = fig.add_subplot(gs0[0, n_channels])
    ax.imshow(height_map, cmap='plasma', vmin=1, vmax=1.5)
    ax.set_ylabel(f'Emission Height')
    ax.yaxis.set_label_position("right")
    ax.set_axis_off()

    ax = fig.add_subplot(gs0[1, n_channels])
    # select index
    y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4 # select point in first quadrant
    plot_ray_sampling(z_vals_stratified[y, x] - distance, z_vals_hierach[y, x] - distance, ax)

    wandb.log({"Comparison": fig})
    plt.close('all')


def log_overview(images, poses, times, wavelengths):
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    colors = cm.get_cmap('viridis')(Normalize()(times))
    # fix arrow heads (2) + shaft color (2) --> 3 color elements
    cs = colors.tolist()
    for c in colors:
        cs.append(c)
        cs.append(c)

    iter_list = list(enumerate(images))
    step = max(1, len(iter_list) // 50)
    for i, img in iter_list[::step]:
        fig = plt.figure(figsize=(16, 8), dpi=150)
        ax = plt.subplot(121, projection='3d')
        # plot all viewpoints
        _ = ax.quiver(
            origins[..., 0].flatten(),
            origins[..., 1].flatten(),
            origins[..., 2].flatten(),
            dirs[..., 0].flatten(),
            dirs[..., 1].flatten(),
            dirs[..., 2].flatten(), color=cs, length=50, normalize=False, pivot='middle',
            linewidth=2, arrow_length_ratio=0.1)

        # plot current viewpoint
        _ = ax.quiver(
            origins[i:i + 1, ..., 0].flatten(),
            origins[i:i + 1, ..., 1].flatten(),
            origins[i:i + 1, ..., 2].flatten(),
            dirs[i:i + 1, ..., 0].flatten(),
            dirs[i:i + 1, ..., 1].flatten(),
            dirs[i:i + 1, ..., 2].flatten(), length=50, normalize=False, color='red', pivot='middle', linewidth=5,
            arrow_length_ratio=0.2)

        d = (1.2 * u.AU).to(u.solRad).value
        ax.set_xlim(-d, d)
        ax.set_ylim(-d, d)
        ax.set_zlim(-d, d)
        ax.scatter(0, 0, 0, marker='o', color='yellow')

        ax = plt.subplot(122)
        # plot corresponding image
        cmap = plt.get_cmap(f'sdoaia{int(wavelengths[i])}').copy()
        ax.imshow(img, norm=sdo_img_norm, cmap=cmap)
        ax.set_axis_off()
        ax.set_title('Time: %s' % unnormalize_datetime(times[i]).isoformat(' '))

        wandb.log({'Overview': fig}, step=i)
        plt.close(fig)


def plot_ray_sampling(
        z_vals: torch.Tensor,
        z_hierarch: Optional[torch.Tensor] = None,
        ax: Optional[np.ndarray] = None):
    r"""
    Plot stratified and (optional) hierarchical samples.
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o', markersize=4)
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o', markersize=4)
    ax.set_ylim([-1, 2])
    # ax.set_xlim([-1.3, 1.3])
    # ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
