import torch
from torch import nn

from sunerf.rendering.base import SuNeRFRendering, cumprod_exclusive


class DensityTemperatureRadiativeTransfer(SuNeRFRendering):

    def __init__(self, model_config=None, **kwargs):
        model_config = {} if model_config is None else model_config
        model_config.update({'d_input': 4, 'd_output': 2, })  # x,y,z,t --> emission, absorption (required model config)
        super().__init__(model_config=model_config, **kwargs)

    def raw2outputs(self, 
		raw: list,
		z_vals: torch.Tensor,
		rays_d: torch.Tensor,
		wavelengths: torch.Tensor
		):
            """
            Convert the raw NeRF output into emission and absorption.
            
            raw: output of NeRF, 2 values per sampled point
            z_vals: distance along the ray as measure from the origin
            """
            wavelengths = wavelengths[:,None,:].repeat(1, z_vals.shape[1], 1)

            # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
            # compute line element (dz) for integration
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists[..., :1], dists], dim=-1)

            # Multiply each distance by the norm of its corresponding direction ray
            # to convert to real world distance (accounts for non-unit directions).
            dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

            # Convert distance into cm to match log_density and log_temperature response
            solRad2cm = (1*u.solRad).to(u.cm).value
            dists = dists*solRad2cm
            
            # Get log log_density and expand to match size of wavelength channels
            # density = torch.float_power(10, nn.functional.relu(raw[0][...,0])+base_rho)
            density = (nn.functional.relu(raw[0][...,0])*self.norm_rho).pow(1/self.pow_rho) + self.base_rho
            density = density[:, :, None].repeat(1,1,wavelengths.shape[2])

            # Get log_temperature and expand to match size of wavelength channels
            log_temperature = nn.functional.relu(raw[0][...,1]) + self.base_temp
            log_temperature = log_temperature[:, :, None].repeat(1,1,wavelengths.shape[2])

            temperature_response = torch.zeros_like(wavelengths) 
            for wavelength in torch.unique(wavelengths):
                if wavelength > 0:
                    wavelength_key = int(wavelength.detach().cpu().numpy().item())
                    tmp_response = self.response[wavelength_key](log_temperature.flatten()).reshape(wavelengths.shape)
                    temperature_response[wavelengths==wavelength] = tmp_response[wavelengths==wavelength].double()


            # Get absorption coefficient
            absortpion_coefficients = torch.zeros_like(wavelengths)
            for wavelength in torch.unique(wavelengths):
                if wavelength > 0:
                    wavelength_key = str(int(wavelength.detach().cpu().numpy().item()))
                    absortpion_coefficients[wavelengths==wavelength] = torch.float_power(10, -(nn.functional.relu(raw[1][wavelength_key]) + self.base_abs))

            # Link to equation: https://www.wolframalpha.com/input?i=df%28z%29%2Fdz+%3D+e%28z%29+-+a%28z%29*f%28z%29%2C+f%280%29+%3D+0

            absorption_integral = density*absortpion_coefficients
            # trapezoid rule
            absorption_integral = 0.5*(absorption_integral[:, :-1, :]+absorption_integral[:, 1:, :])*dists[:,1:,None]
            # cumulative sum
            absorption_integral = -torch.cumsum(absorption_integral, dim=1)

            emission_integral = density.pow(2)*temperature_response
            # use midpoint
            emission_integral = 0.5*(emission_integral[:, :-1, :]+emission_integral[:, 1:, :])
            # Compose the term
            emission_integral = torch.exp(-absorption_integral)*emission_integral
            # trapezoid rule
            emission_integral = 0.5*(emission_integral[:, :-1,:]+emission_integral[:, 1:,:])*dists[:,1:-1,None]
            # TODO: Check which dists should go here: dists[:,1:-1] or dists[:,2:]

            pixel_intensity = torch.exp(absorption_integral[:,-1,:])*emission_integral.sum(1)*raw[2]

            for wavelength in torch.unique(wavelengths):
                if wavelength > 0:
                    wavelength_key = int(wavelength.detach().cpu().numpy().item())
                    pixel_intensity[wavelengths[:,0,:]==wavelength] = self.channel_norms[wavelength_key](pixel_intensity[wavelengths[:,0,:]==wavelength])
                
            # set the weigths to the intensity contributions
            weights = (nn.functional.relu(raw[0][...,0])*self.norm_rho).pow(1/self.pow_rho) + self.base_rho
            weights = weights / (weights.sum(1)[:, None] + 1e-10)

            return {'image': pixel_intensity, 'weights': weights, 'absorption': absorption}
