import torch
from torch import nn

from sunerf.model.model import EmissionModel
from sunerf.rendering.base_tracing import SuNeRFRendering, cumprod_exclusive


class EmissionRadiativeTransfer(SuNeRFRendering):

    def __init__(self, model_config=None, **kwargs):
        model_config = {} if model_config is None else model_config
        # setup models
        coarse_model = EmissionModel(1, **model_config)
        fine_model = EmissionModel(1, **model_config)
        super().__init__(coarse_model=coarse_model, fine_model=fine_model, **kwargs)

    def raw2outputs(self, raw: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor, **kwargs):
        r"""
        Convert the raw NeRF output into emission and absorption.

        raw: output of NeRF, 2 values per sampled point
        z_vals: distance along the ray as measure from the origin
        """

        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        # compute line element (dz) for integration
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists[..., :1], dists], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # emission ([..., 0]; epsilon(z)) and absorption ([..., 1]; kappa(z)) coefficient per unit volume
        # dtau = - kappa dz
        # I' / I = - kappa dz --> I' emerging intensity; I incident intensity;
        emission = raw['emission'][..., 0] # torch.exp(raw[..., 0])
        intensity = emission * dists  # emission per sampled point [n_rays, n_samples]

        # transmission per sampled point [n_rays, n_samples]
        alpha = raw['alpha'][..., 0] #nn.functional.relu(raw[..., 1])
        absorption = torch.exp(-alpha * dists)
        # [1, .9, 1, 0, 0, 1] --> less dense objects transmit light (1); dense objects absorbe light (0)

        # compute total absorption for each light ray (intensity)
        # how much light is transmitted from each sampled point
        # first intensity has no absorption (1, t[0], t[0] * t[1], t[0] * t[1] * t[2], ...)
        total_absorption = cumprod_exclusive(absorption + 1e-10)
        # [(1), 1, .9, .9, 0, 0] --> total absorption for each point along the ray
        # apply absorption to intensities
        emerging_intensity = intensity * total_absorption  # integrate total intensity [n_rays, n_samples - 1]
        # sum all intensity contributions
        pixel_intensity = emerging_intensity.sum(1)[:, None]

        # set the weigths to the intensity contributions (sample primary contributing regions)
        weights = emerging_intensity
        weights = weights / (weights.sum(1)[:, None] + 1e-10)

        mean_absorption = (1 - absorption).mean(1)

        return {'image': pixel_intensity, 'weights': weights, 'absorption_map': mean_absorption}
