import torch
from torch import nn
from astropy import units as u
from sunpy.io.special import read_genx
from xitorch.interpolate import Interp1D
from sunerf.rendering.base_tracing import SuNeRFRendering, cumprod_exclusive
from scipy.io import readsav
import numpy as np

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
    prod_expo = torch.cumprod(integrand, dim=0)  # product of exponentials
    return prod_expo



class DensityTemperatureRadiativeTransfer(SuNeRFRendering):
    """ SuNeRF model for rendering filtergrams from density and temperature fields.

    Parameters
    ----------
    wavelengths : list
        List of wavelengths to render filtergrams.
    model_config : dict
        Configuration of the model.
    device : torch.device
        Device to run the model.
    aia_exp_time : float
        Typical AIA exposure time.
    pixel_intensity_factor : float
        Factor to scale the pixel intensity.

    Returns
    -------
    SuNeRF model for rendering filtergrams.
    """

    def __init__(self, model_config=None, device=None, aia_exp_time=2.9, 
                 temperature_response_normalization = {0: 1e17, 1: 1e-10, 2: 1e-10}, use_aia_tresp = False, **kwargs):
        """ Initialize the DensityTemperatureRadiativeTransfer model.

        Parameters
        ----------
        wavelengths : list
            List of wavelengths to render filtergrams.
        model_config : dict
            Configuration of the model.
        device : torch.device
            Device to run the model.
        aia_exp_time : float
            Typical AIA exposure time.
        pixel_intensity_factor : float
            Factor to scale the pixel intensity.

        Returns
        -------
        DensityTemperatureRadiativeTransfer model.
        """

        # Initialize the model
        model_config = {} if model_config is None else model_config
        super().__init__(model_config=model_config, **kwargs)

        # Initialize the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device

        # Read the AIA temperature response functions
        aia_resp = read_genx("sunerf/data/aia_temp_resp.genx")
        self.response = {}
        self.response['0_wavelength_names'] = torch.Tensor([94, 131, 171, 193, 211, 304, 335])
        self.response['1_wavelength_names'] = torch.Tensor([171, 195, 284, 304])
        self.response['2_wavelength_names'] = torch.Tensor([171, 195, 284, 304])

        # Loop over the AIA temperature response functions
        for key in aia_resp.keys():
            # Skip the header
            if key != 'HEADER':
                # Get the wavelength
                wavelength = int(key[1:])
                # Get the log temperature
                log_temperature = aia_resp[f'A{wavelength}']['LOGTE']
                # Get the response and multiply by the typical AIA exposure time
                response = aia_resp[f'A{wavelength}']['TRESP'] * aia_exp_time * temperature_response_normalization[0]
                # Interpolate the temperature response function
                self.response[f'0_{wavelength}_LOGTE'] = torch.Tensor(log_temperature)
                self.response[f'0_{wavelength}_TRESP'] = torch.Tensor(response)

                if use_aia_tresp:
                    wavelength_dict = wavelength
                    if wavelength == 193:
                        wavelength_dict = 195
                    if wavelength == 211:
                        wavelength_dict = 284

                    self.response[f'1_{wavelength_dict}_LOGTE'] = torch.Tensor(log_temperature)
                    self.response[f'1_{wavelength_dict}_TRESP'] = torch.Tensor(response)

                    self.response[f'2_{wavelength_dict}_LOGTE'] = torch.Tensor(log_temperature)
                    self.response[f'2_{wavelength_dict}_TRESP'] = torch.Tensor(response)                                                       

        # Loop over the EUVIA temperature response functions
        euvia_resp = readsav('sunerf/data/euvi_response/ahead_sre_chianti2_fludra_mazzotta_001.geny')
        for i, key in enumerate (euvia_resp['p0'].item(0)[8][0:4, 0]):
            # Get the wavelength
            wavelength = int(key)
            # Get the log temperature
            log_temperature = np.log10(euvia_resp['p0'].item(0)[5]).astype(np.float32)
            # Get the response and multiply by the typical AIA exposure time
            response = euvia_resp['p0'].item(0)[7][i, 0, :].astype(np.float32) * temperature_response_normalization[1]
            # Interpolate the temperature response function
            self.response[f'1_{wavelength}_LOGTE'] = torch.Tensor(log_temperature)
            self.response[f'1_{wavelength}_TRESP'] = torch.Tensor(response)

        # Loop over the EUVIB temperature response functions
        euvib_resp = readsav('sunerf/data/euvi_response/behind_sre_chianti2_fludra_mazzotta_001.geny')
        for i, key in enumerate (euvib_resp['p0'].item(0)[8][0:4, 0]):
            # Get the wavelength
            wavelength = int(key)
            # Get the log temperature
            log_temperature = np.log10(euvib_resp['p0'].item(0)[5]).astype(np.float32)
            # Get the response and multiply by the typical AIA exposure time
            response = euvia_resp['p0'].item(0)[7][i, 0, :].astype(np.float32) * temperature_response_normalization[2]
            # Interpolate the temperature response function
            self.response[f'2_{wavelength}_LOGTE'] = torch.Tensor(log_temperature)
            self.response[f'2_{wavelength}_TRESP'] = torch.Tensor(response)

        self.response = nn.ParameterDict(self.response) 
            

    def _render(self, model, query_points, rays_d, rays_o, z_vals, wavelengths, instruments):
        """ Render the filtergrams from density and temperature fields.

        Parameters
        ----------
        model : nn.Module
            DensityTemperature model.
        query_points : torch.Tensor
            Query points.
        rays_d : torch.Tensor
            Ray directions.
        rays_o : torch.Tensor
            Ray origins.
        z_vals : torch.Tensor
            Z values. Position of the samples along the rays.
        wavelengths : torch.Tensor
            Wavelengths.
        instruments : torch.Tensor
            Instruments.

        Returns
        -------
        out : dict
            Rendered filtergrams. Contains the following:
            - image: Rendered filtergrams.
            - weights: Weights of the filtergrams.
            - absorption: Absorption coefficient.
        """

        # Get the shape of the query points
        query_points_shape = query_points.shape[:-1]
        # Flatten the query points
        flat_query_points = query_points.view(-1, 4)
        # Get the model output at the query points
        state = model.forward(flat_query_points)

        # Save the model output and parameters in the state dictionary
        state['inferences'] = state['inferences'].reshape(*query_points_shape, state['inferences'].shape[-1])
        state['z_vals'] = z_vals
        state['rays_d'] = rays_d
        state['wavelengths'] = wavelengths
        state['instruments'] = instruments
        # TODO: Make sure emission & absorption work
        # Perform differentiable volume rendering to re-synthesize the filtergrams.
        # state = {'raw': raw, 'z_vals': z_vals, 'rays_d': rays_d}  
        out = self.raw2outputs(**state)
        # out contains the rendered filtergrams, the weights of the filtergrams and the absorption coefficient
        return out
    
    def raw2outputs(self, inferences: torch.Tensor, log_abs: torch.Tensor, vol_c: torch.Tensor, z_vals: torch.Tensor,
                    rays_d: torch.Tensor, wavelengths: torch.Tensor, instruments: torch.Tensor):
        """ Convert the raw NeRF output into emission and absorption.

        Parameters
        ----------
        rho_T : torch.Tensor
            Raw NeRF output.
        log_abs : nn.ParameterDict
            Log absorption coefficient.
        vol_c : torch.Tensor
            Volume of the cell.
        z_vals : torch.Tensor
            Z values. Position of the samples along the rays.
        rays_d : torch.Tensor
            Ray directions.
        wavelengths : torch.Tensor
            Wavelengths.
        instruments : torch.Tensor
            Instruments.

        Returns
        -------
        out : dict
            Rendered filtergrams. Contains the following:
            - image: Rendered filtergrams.
            - weights: Weights of the filtergrams.
            - absorption: Absorption coefficient.
        """
        wavelengths = wavelengths[:, None, :].expand(wavelengths.shape[0], inferences.shape[1], wavelengths.shape[1])
        instruments = instruments[:, None, None].expand(instruments.shape[0], inferences.shape[1], wavelengths.shape[2])

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
        density = torch.exp(inferences[...,0])
        density = density[:, :, None].expand(density.shape[0], density.shape[1], wavelengths.shape[2])

        # Get log_temperature and expand to match size of wavelength channels
        log_temperature = inferences[...,1]
        log_temperature = log_temperature[:, :, None].expand(log_temperature.shape[0], log_temperature.shape[1], wavelengths.shape[2])

        temperature_response = torch.zeros_like(log_temperature)
        for instrument in torch.unique(instruments):
            for wavelength in torch.unique(wavelengths[instruments==instrument]):
                if wavelength > -1:
                    wavelength_key = int(self.response[f'{int(instrument)}_wavelength_names'][int(wavelength)])
                    response = Interp1D(self.response[f'{int(instrument)}_{wavelength_key}_LOGTE'],
                                        self.response[f'{int(instrument)}_{wavelength_key}_TRESP'],
                                        method='linear', extrap=0)
                    tmp_response = response(log_temperature.flatten()).reshape(temperature_response.shape)
                    mask = torch.logical_and(wavelengths==wavelength, instruments==instrument)
                    temperature_response[mask] = tmp_response[mask]


        # Get absorption coefficient
        absorption_coefficients = torch.zeros_like(log_temperature)
        for instrument in torch.unique(instruments):
            for wavelength in torch.unique(wavelengths[instruments==instrument]):
                if wavelength > -1:
                    mask = torch.logical_and(wavelengths==wavelength, instruments==instrument)
                    absorption_coefficients[mask] = torch.float_power(10, -(nn.functional.relu(log_abs[wavelength,instrument]))).float() # removed base_abs
                    absorption_coefficients[mask] = nn.functional.relu(log_abs[wavelength,instrument]) # removed base_abs

        # Link to equation:
        # https://www.wolframalpha.com/input?i=df%28z%29%2Fdz+%3D+e%28z%29+-+a%28z%29*f%28z%29%2C+f%280%29+%3D+0
        absorption = density*absorption_coefficients
        absorption_integral = torch.cumulative_trapezoid(absorption, x=z_vals[:,:,None], dim=1)

        emission = density.pow(2)*temperature_response
        pixel_intensity_term = torch.exp(-absorption_integral) * emission[:,0:-1,:]   #TODO: Check which emission indexes should go here
        pixel_intensity = torch.trapezoid(pixel_intensity_term, x=z_vals[:, 0:-1, None], dim=1)
        for instrument in torch.unique(instruments):
            mask = instruments[:, 0, :] == instrument
            pixel_intensity[mask] = pixel_intensity[mask] * vol_c[instrument]   # TODO: Check which z_vals indexes should go here

        # set the weights to the intensity contributions
        weights = (nn.functional.relu(inferences[...,0]))
        weights = weights / (weights.sum(1)[:, None] + 1e-10)

        return {'image': pixel_intensity, 'weights': weights, 'regularizing_quantity': nn.functional.relu(inferences[...,0])} # density is the regularizing quantity
    
    def regularization(self, distance, regularizing_quantity):
        return torch.relu(distance[:,:] - 1.25 / self.Rs_per_ds) * torch.relu(regularizing_quantity)
