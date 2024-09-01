from typing import Optional

import numpy as np
import torch
import wandb
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from pytorch_lightning import Callback
from skimage.metrics import structural_similarity

from sunerf.data.date_util import unnormalize_datetime
from sunerf.data.utils import sdo_img_norm


class BaseCallback(Callback):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_validation_outputs(self, pl_module):
        if self.name not in pl_module.validation_outputs:
            return None
        outputs = pl_module.validation_outputs[self.name]
        return outputs


class TestImageCallback(BaseCallback):

    def __init__(self, name, image_shape, cmap='gray'):
        super().__init__(name)
        self.image_shape = image_shape
        self.cmap = plt.get_cmap(cmap)
        self.normalize = ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005), clip=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        # reshape
        outputs = {k: v.view(*self.image_shape, *v.shape[1:]).cpu().numpy() for k, v in outputs.items()}

        fine_image = self.normalize(outputs['fine_image'])
        target_image = self.normalize(outputs['target_image'])
        coarse_image = self.normalize(outputs['coarse_image'])

        plot_samples(fine_image, coarse_image, outputs['height_map'], outputs['absorption_map'],
                     target_image, outputs['z_vals_stratified'], outputs['z_vals_hierarchical'],
                     outputs['distance'].mean(), self.cmap)

        val_loss = ((fine_image - target_image) ** 2).mean()
        val_ssim = structural_similarity(target_image[..., 0], fine_image[..., 0], data_range=1)
        val_psnr = -10. * np.log10(val_loss)

        wandb.log({'validation.loss': val_loss, 'validation.ssim': val_ssim, 'validation.psnr': val_psnr})

class TestMultiThermalImageCallback(BaseCallback):

    def __init__(self, name, image_shape, wavelengths, instrument):
        super().__init__(name)
        self.image_shape = image_shape
        self.wavelengths = wavelengths
        if instrument == 0:
            self.wavelength_cmap = np.array([94, 131, 171, 193, 211, 304, 335])
        elif instrument == 1:
            self.wavelength_cmap = np.array([171, 193, 211, 304])
        elif instrument == 2:
            self.wavelength_cmap = np.array([171, 193, 211, 304])
        else:
            self.wavelength_cmap = np.array([94, 131, 171, 193, 211, 304, 335])       

    def on_validation_epoch_end(self, trainer, pl_module):

        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        # reshape
        outputs = {k: v.view(*self.image_shape, *v.shape[1:]).cpu().numpy() for k, v in outputs.items()}

        fine_image = outputs['fine_image']
        target_image = outputs['target_image']
        coarse_image = outputs['coarse_image']

        plot_samples_multithermal(fine_image, coarse_image, outputs['height_map'], 
                     target_image, outputs['z_vals_stratified'], outputs['z_vals_hierarchical'], self.wavelengths,
                     self.wavelength_cmap,
                     outputs['distance'].mean())

        val_loss = ((fine_image - target_image) ** 2).mean()
        val_ssim = structural_similarity(target_image[..., 0], fine_image[..., 0], data_range=1)
        val_psnr = -10. * np.log10(val_loss)

        wandb.log({'validation.loss': val_loss, 'validation.ssim': val_ssim, 'validation.psnr': val_psnr})
        


def plot_samples(fine_image, coarse_image, height_map, absorption_map, target_image, z_vals_stratified,
                 z_vals_hierach, distance, cmap):
    # Log example images on wandb
    # # Plot example outputs

    fig, ax = plt.subplots(1, 6, figsize=(30, 4))

    ax[0].imshow(target_image[..., 0], cmap=cmap, norm=sdo_img_norm)
    ax[0].set_title(f'Target')
    ax[1].imshow(fine_image[..., 0], cmap=cmap, norm=sdo_img_norm)
    ax[1].set_title(f'Fine')
    ax[2].imshow(coarse_image[..., 0], cmap=cmap, norm=sdo_img_norm)
    ax[2].set_title(f'Coarse')
    ax[3].imshow(height_map, cmap='plasma', vmin=1, vmax=1.3)
    ax[3].set_title(f'Emission Height')
    ax[4].imshow(absorption_map, cmap='viridis', vmin=0)
    ax[4].set_title(f'Absorption')

    # select index
    y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4  # select point in first quadrant
    plot_ray_sampling(z_vals_stratified[y, x] - distance, z_vals_hierach[y, x] - distance, ax[-1])

    wandb.log({"Comparison": fig})
    plt.close('all')

def plot_samples_multithermal(fine_image, coarse_image, height_map,
                 target_image,
                 z_vals_stratified,
                 z_vals_hierach,
                 wavelengths,
                 wavelength_cmap,
                 distance
                 ):
    # Log example images on wandb
    # # Plot example outputs

    n_channels = wavelengths.shape[0]
    wavelength_cmap = wavelength_cmap[wavelengths]

    fig = plt.figure(figsize=2*np.array([n_channels+1, 3]), dpi=500)
    gs0 = fig.add_gridspec(3, n_channels+1, wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    for i in np.arange(0,n_channels):
        normalize = ImageNormalize(vmin=np.min(target_image[..., i]), vmax=np.max(target_image[..., i]), stretch=AsinhStretch(0.005), clip=True)
        cmap = plt.get_cmap(f'sdoaia{int(wavelength_cmap[i])}').copy()
        ax = fig.add_subplot(gs0[0, i])
        # ax.imshow(testimg[..., i], cmap=cmap, norm=sdo_img_norm)
        ax.imshow(normalize(target_image[..., i]), cmap=cmap, norm=sdo_img_norm)
        if i==0:
            ax.set_ylabel(f'Target')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs0[1, i])
        ax.imshow(normalize(fine_image[..., i]), cmap=cmap, norm=sdo_img_norm)
        if i==0:
            ax.set_ylabel(f'Prediction')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs0[2, i])
        ax.imshow(normalize(coarse_image[..., i]), cmap=cmap, norm=sdo_img_norm)
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


def log_overview(images, poses, times, cmap, seconds_per_dt, ref_time, wavelengths=None):
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
    colors = plt.get_cmap('viridis')(Normalize()(times))
    # fix arrow heads (2) + shaft color (2) --> 3 color elements
    cs = colors.tolist()
    for c in colors:
        cs.append(c)
        cs.append(c)

    norm = ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005), clip=True)

    iter_list = list(enumerate(images))
    step = max(1, len(iter_list) // 20)
    for i, img in iter_list[::step]:
        fig = plt.figure(figsize=(16, 8), dpi=100)
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
        if cmap is not None:
            ax.imshow(img, norm=norm, cmap=cmap)
        if wavelengths is not None:
            cmap = plt.get_cmap(f'sdoaia{int(wavelengths[i])}').copy()
            ax.imshow(img, norm=norm, cmap=cmap)
        ax.set_axis_off()
        ax.set_title('Time: %s' % unnormalize_datetime(times[i], seconds_per_dt, ref_time).isoformat(' '))

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
    ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
