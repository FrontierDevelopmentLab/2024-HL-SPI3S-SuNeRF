import numpy as np
import torch
from torch import nn
import astropy.units as u
import matplotlib.pyplot as plt
import sunpy

# from s4pi.maps.train.heliocentric_cartesian_transformation import pose_spherical
# from s4pi.maps.train.parallel_ray_sampling import get_rays
# from s4pi.maps.train.coordinate_transformation import pose_spherical
# from s4pi.maps.train.ray_sampling import get_rays
# from s4pi.maps.train.sampling import sample_non_uniform_box as sample_stratified
# from s4pi.maps.train.volume_render import volume_render

class SimpleStar(nn.Module):
    r""" Simple star that pretends to be a fully trained nerf
    """
    def __init__(self, h0=60*u.Mm, t0=1.4e6*u.K, R_s=1.02*u.solRad, t_photosphere = 5777*u.K,
                                    rho_0 = 3.0e8/u.cm**3, base_rho=3.0, base_temp=5.5, base_abs=16.0):
        r"""
        Arguments
        ----------
        main_args : string
            path to hyperperams.yaml file with training configuration
        channel_norms: dict
            normalization for all channels
        aia_response_path : string
            path to file containing SDO/AIA temperature resp. functions
        """
        super().__init__()
        self.h0 = h0
        self.t0 = t0
        self.R_s = R_s
        self.t_photosphere = t_photosphere
        self.rho_0 = rho_0
        self.base_rho = base_rho
        self.base_temp = base_temp
        self.base_abs = base_abs   
                 

    def forward(self, query_points):
        """Translates x,y,z position and time into a density and temperature map.
            See (Pascoe et al. 2019) https://iopscience.iop.org/article/10.3847/1538-4357/ab3e39

        Args:
            batch (float,vector): Batch of points to render
        Params:
            h0 (float, optional): isothermal scale height. Defaults to 60*u.Mm.
            t0 (_type_, optional): coronal temperature. Defaults to 1.2*u.MK.
            R_s (_type_, optional): Effective solar surface radius. Defaults to 1.2*u.solRad.
            t_photosphere (float, optional): Temperature at the solar surface. Defaults to 5777*u.K.
            rho_0 (float): Density at the solar surface. Defaults to 2e8/u.cm**3.
        Returns:
            rho (float): Density at x,y,z,t
            temp (float): Temperature at x,y,z,t
        """

        x = query_points[:,0]
        y = query_points[:,1]
        z = query_points[:,2]

        # Radius (distance from the center of the sphere) in solar radii
        radius = torch.sqrt(x**2 + y**2 + z**2)
        # Initialize density and temperature
        rho = torch.zeros_like(radius)
        temp = torch.zeros_like(radius)

        #find indices of radii less than solRad and greater than solRad
        less_than_index = radius <= 1.0
        else_index = radius > 1.0
        
        # If radius is less then 1 solar radii...
        rho[less_than_index] = self.rho_0.value
        # If radius is greater than 1 solar radii...
        rho[else_index] = self.rho_0.value * torch.exp(1/self.h0.to(u.solRad).value*(1/radius[else_index]-1)) #See equation 4 in Pascoe et al. 2019
        rho = torch.log(rho)

        # Simple temperature model (depends on radius)
        # If radius is less then 1 solar radii...
        temp[less_than_index] = self.t_photosphere.value
        # If radius is between 1 solar radii and R_s solar radii...
        R_s_index = torch.logical_and(radius > 1, radius <= self.R_s.value)
        temp[R_s_index] = (radius[R_s_index]-1)*((self.t0-self.t_photosphere).value/(self.R_s.value - 1))+ self.t_photosphere.value #See equation 6 in Pascoe et al. 2019
        # If radius is greater than R_s solar radii, use constant...
        out_sun_index = radius > self.R_s.value
        temp[out_sun_index]= self.t0.value
        temp = torch.log10(temp)

        log_absortpion = nn.ParameterDict([
                                        ['94',  torch.tensor(20.4, dtype=torch.float32)],
                                        ['131', torch.tensor(20.2, dtype=torch.float32)],
                                        ['171', torch.tensor(20.0, dtype=torch.float32)],
                                        ['193', torch.tensor(19.8, dtype=torch.float32)],
                                        ['211', torch.tensor(19.6, dtype=torch.float32)],
                                        ['304', torch.tensor(19.4, dtype=torch.float32)],
                                        ['335', torch.tensor(19.2, dtype=torch.float32)]
                                ])

        volumetric_constant = torch.tensor(25.0)

        # Output density, temperature, absortpion and volumetric constant
        return {'rho_T': torch.stack((rho,temp), dim=-1), 'log_abs': log_absortpion, 'vol_c': volumetric_constant} # removed to model device


