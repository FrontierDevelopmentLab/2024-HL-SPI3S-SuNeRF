import torch
from torch import nn
from astropy import units as u
from sunpy.io.special import read_genx
from xitorch.interpolate import Interp1D

from sunerf.rendering.base_tracing import SuNeRFRendering, cumprod_exclusive

def rectangular_integral(x, y, cumsum=False):
    '''
    Compute the rectangular rule integration of a function y(x) given x

    Parameters:
    x: torch.tensor
        x values
    y: torch.tensor
        y values
    cumsum: bool    
        If True, return the cumulative sum of the integral

    Returns:
    rectangular_integration: torch.tensor
        Rectangular rule integration of y(x)
    '''
    dist = x[1:] - x[0:-1]
    integrand = dist*y[:-1]
    
    if cumsum:
        rectangular_integration = torch.cumsum(integrand, dim=0)
    else:
        rectangular_integration = torch.sum(integrand)
    
    return rectangular_integration


def prod_exponential_rectangular(x, y):
    '''
    Compute the product of exponentials of a function y(x) given x

    Parameters:
    x: torch.tensor
        x values
    y: torch.tensor
        y values

    Returns:
    prod_expo: torch.tensor
        Product of exponentials of y(x)
    '''
    
    dist = x[1:] - x[0:-1]
    integrand = dist*y[:-1]
    integrand = torch.exp(-integrand)
    prod_expo = torch.cumprod(integrand, dim=0) # product of exponentials
    return prod_expo

def prod_exponential_trapezoid(x, y):
    '''
    Compute the product of exponentials of a function y(x) given x

    Parameters:
    x: torch.tensor
        x values
    y: torch.tensor
        y values

    Returns:
    prod_expo: torch.tensor
        Product of exponentials of y(x)
    '''
    
    dist = x[1:] - x[0:-1]
    integrand = dist*(y[1:] + y[0:-1]) / 2
    integrand = torch.exp(-integrand)
    prod_expo = torch.cumprod(integrand, dim=0) # product of exponentials
    return prod_expo

class DensityTemperatureRadiativeTransfer(SuNeRFRendering):

    def __init__(self, wavelengths, model_config=None, device=None, aia_exp_time=2.9, pixel_intensity_factor=1e10, **kwargs):
        model_config = {} if model_config is None else model_config
        super().__init__(model_config=model_config, **kwargs)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device
        self.wavelengths = torch.tensor(wavelengths).to(device)
        self.pixel_intensity_factor = pixel_intensity_factor
    
        aia_resp = read_genx("sunerf/data/aia_temp_resp.genx")
        self.response = {}
        for key in aia_resp.keys():
            if key != 'HEADER':
                wavelength = int(key[1:])
                log_temperature = aia_resp[f'A{wavelength}']['LOGTE']  #log_temperature
                response = aia_resp[f'A{wavelength}']['TRESP']*aia_exp_time  # multiply response by typical AIA exposure time
                self.response[wavelength] = Interp1D(torch.from_numpy(log_temperature).float().to(self.device), torch.from_numpy(response).float().to(self.device), method='linear', extrap=0)


    def _render(self, model, query_points, rays_d, rays_o, z_vals):
        query_points_shape = query_points.shape[:-1]
        flat_query_points = query_points.view(-1, 4)
        state = model(flat_query_points)
        
        state['rho_T'] = state['rho_T'].reshape(*query_points_shape, state['rho_T'].shape[-1])
        state['z_vals'] = z_vals
        state['rays_d'] = rays_d
        state['wavelengths'] = self.wavelengths
        # Perform differentiable volume rendering to re-synthesize the filtergrams.
        #state = {'raw': raw, 'z_vals': z_vals, 'rays_d': rays_d} #TODO include wavelenghts
        out = self.raw2outputs(**state)
        return out
    
    def raw2outputs(self, 
		rho_T: torch.Tensor,
        log_abs: nn.ParameterDict,
        vol_c: torch.Tensor,
		z_vals: torch.Tensor,
		rays_d: torch.Tensor,
		wavelengths: torch.Tensor
		):
            """
            Convert the raw NeRF output into emission and absorption.
            
            raw: output of NeRF, 2 values per sampled point
            z_vals: distance along the ray as measure from the origin
            """
            wavelengths = wavelengths[None, None, :] # TODO: Fix broadcasting later for multi wavelength

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
            # density = (nn.functional.relu(rho_T[...,0])*self.norm_rho).pow(1/self.pow_rho) + self.base_rho
            density = torch.exp(nn.functional.relu(rho_T[...,0]))
            density = density[:, :, None].repeat(1,1,wavelengths.shape[2])

            # Get log_temperature and expand to match size of wavelength channels
            log_temperature = nn.functional.relu(rho_T[...,1])
            log_temperature = log_temperature[:, :, None].repeat(1,1,wavelengths.shape[2])

            temperature_response = torch.zeros_like(log_temperature) 
            for wavelength in torch.unique(wavelengths):
                if wavelength > 0:
                    wavelength_key = int(wavelength.detach().cpu().numpy().item())
                    tmp_response = self.response[wavelength_key](log_temperature.flatten()).reshape(temperature_response.shape)
                    temperature_response[wavelengths*torch.ones_like(temperature_response)==wavelength] = tmp_response[wavelengths*torch.ones_like(temperature_response)==wavelength]

            # Get absorption coefficient
            absortpion_coefficients = torch.zeros_like(wavelengths).float()
            for wavelength in torch.unique(wavelengths):
                if wavelength > 0:
                    wavelength_key = str(int(wavelength.detach().cpu().numpy().item()))
                    absortpion_coefficients[wavelengths==wavelength] = torch.float_power(10, -(nn.functional.relu(log_abs[wavelength_key]))).float() # removed base_abs

            # Link to equation: https://www.wolframalpha.com/input?i=df%28z%29%2Fdz+%3D+e%28z%29+-+a%28z%29*f%28z%29%2C+f%280%29+%3D+0
            
            absorption = density*absortpion_coefficients
            absorption_integral = torch.cumulative_trapezoid(absorption, x=z_vals[:,:,None], dim=1)

            emission = density.pow(2)*temperature_response
            pixel_intensity_term = torch.exp(-absorption_integral) * emission[:,1:,:]   #TODO: Check which emission indexes should go here
            pixel_intensity = torch.trapezoid(pixel_intensity_term, x=z_vals[:, 1:, None], dim=1) * vol_c * self.pixel_intensity_factor   # TODO: Check which z_vals indexes should go here

            # set the weigths to the intensity contributions
            weights = (nn.functional.relu(rho_T[...,0]))
            weights = weights / (weights.sum(1)[:, None] + 1e-10)

            return {'image': pixel_intensity, 'weights': weights, 'absorption': absorption}
