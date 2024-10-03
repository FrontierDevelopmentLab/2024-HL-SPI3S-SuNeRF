import torch
from torch import nn

from sunerf.model.model import EmissionModel, PlasmaModel
from sunerf.rendering.base_tracing import SuNeRFRendering, cumprod_exclusive


class PlasmaRadiativeTransfer(SuNeRFRendering):

    def __init__(self, model_config=None, **kwargs):
        model_config = {} if model_config is None else model_config
        coarse_model = PlasmaModel(**model_config)
        fine_model = PlasmaModel(**model_config)
        super().__init__(coarse_model=coarse_model, fine_model=fine_model, **kwargs)

    def raw2outputs(self, raw: dict, z_vals: torch.Tensor, rays_d: torch.Tensor,
                    temperature_response: nn.Module, instrument_scaling: nn.Parameter, absorption_model: nn.Module,
                    **kwargs):
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
        dists = dists[:, :, None]

        log_T = raw['log_T']
        log_ne = raw['log_ne']
        T = raw['T']
        ne = raw['ne']

        # intensity = raw['emission']
        # alpha = raw['alpha']
        # ne = intensity
        # log_T = intensity

        # find channel response for temperature and weight by electron density squared
        # assume dirac delta function for T distribution
        response = temperature_response(log_T)
        intensity = 10 ** (response + 2 * log_ne + instrument_scaling)
        #
        # # learn absorption based on electron density
        absorption_input = torch.cat([log_ne, log_T], dim=-1)
        log_nu = absorption_model(absorption_input)['log_nu']
        alpha = 10 ** (log_nu + 2 * log_ne) # alpha = nu * ne^2 = 10^log_nu * 10^( log_ne)^2 = 10^(log_nu + 2 * log_ne)
        alpha = log_ne * 0.0 # ignore absorption for now

        # transmission per sampled point [n_rays, n_samples]
        absorption = torch.exp(-alpha * dists)
        # [1, .9, 1, 0, 0, 1] --> less dense objects transmit light (1); dense objects absorbe light (0)

        # compute total absorption for each light ray (intensity)
        # how much light is transmitted from each sampled point
        # first intensity has no absorption (1, t[0], t[0] * t[1], t[0] * t[1] * t[2], ...)
        integrated_absorption = cumprod_exclusive(absorption + 1e-10, dim=1)
        # [(1), 1, .9, .9, 0, 0] --> total absorption for each point along the ray
        # apply absorption to intensities
        emerging_intensity = intensity * integrated_absorption  # integrate total intensity [n_rays, n_samples - 1]
        # sum all intensity contributions
        integrated_intensity = (emerging_intensity * dists).sum(1)

        # set the weigths to the intensity contributions (sample primary contributing regions)
        weights = emerging_intensity.mean(-1)
        weights = weights / (weights.sum(1)[:, None] + 1e-10)

        total_density = ne.sum(1)
        mean_T = (ne * log_T).sum(1) / total_density
        mean_absorption = (1 - absorption).mean((1, -1))

        # print('MIN/MAX log T', log_T.min(), log_T.max(), log_T.shape)
        # print('MIN/MAX log ne', log_ne.min(), log_ne.max(), log_ne.shape)
        # print('MIN/MAX TR', response.min(), response.max(), response.shape)
        # print('MIN/MAX INTENSITY', intensity.min(), intensity.max(), intensity.shape)
        # print('MIN/MAX INTEGRATED ABSORPTION', integrated_absorption.min(), integrated_absorption.max(),
        #       integrated_absorption.shape)
        # print('MIN/MAX INTEGRATED INTENSITY', integrated_intensity.min(), integrated_intensity.max(),
        #       integrated_intensity.shape)

        return {'image': integrated_intensity, 'weights': weights, 'mean_absorption': mean_absorption,
                'mean_T': mean_T, 'total_ne': total_density}
