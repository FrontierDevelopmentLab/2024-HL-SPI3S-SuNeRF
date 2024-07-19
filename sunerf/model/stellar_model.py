import torch
from torch import nn
import astropy.units as u

class SimpleStar(nn.Module):
    r""" Simple star that pretends to be a fully trained nerf
    """
    def __init__(self, h0:u=60*u.Mm, T0:u=1.4e6*u.K, R_s:u=1.02*u.solRad, t_photosphere:u = 5777*u.K,
                                    rho_0:u = 3.0e8/u.cm**3):
        """_summary_

        Parameters
        ----------
        h0 : u, optional
            Scale height of stellar atmosphere for density stratification, by default 60*u.Mm
        T0 : u, optional
            Temperature of the stellar corona, by default 1.4e6*u.K
        R_s : u, optional
            Radius above which the stellar corona becomes isothermal, by default 1.02*u.solRad
        t_photosphere : u, optional
            Temperature at the photosphere, by default 5777*u.K
        rho_0 : u, optional
            Density at the photosphere, by default 3.0e8/u.cm**3
        """            
        super().__init__()
        self.unit_of_length = u.cm
        self.h0 = h0.to(u.solRad).value
        self.T0 = T0.to(u.K).value
        self.R_s = R_s.to(u.solRad).value
        self.t_photosphere = t_photosphere.to(u.K).value
        self.rho_0 = rho_0.to(1/self.unit_of_length/self.unit_of_length/self.unit_of_length).value

        self.log_absortpion = nn.ParameterDict([
                                ['94',  torch.tensor(20.4, dtype=torch.float32)],
                                ['131', torch.tensor(20.2, dtype=torch.float32)],
                                ['171', torch.tensor(20.0, dtype=torch.float32)],
                                ['193', torch.tensor(19.8, dtype=torch.float32)],
                                ['211', torch.tensor(19.6, dtype=torch.float32)],
                                ['304', torch.tensor(19.4, dtype=torch.float32)],
                                ['335', torch.tensor(19.2, dtype=torch.float32)]
                        ])

        self.stellar_parameters = nn.ParameterDict([
                                ['Rs', torch.tensor(self.R_s, dtype=torch.float32)],
                                ['h0', torch.tensor(self.h0, dtype=torch.float32)],
                                ['T0', torch.tensor(self.T0, dtype=torch.float32)],
                                ['rho_0', torch.tensor(self.rho_0, dtype=torch.float32)]
                        ])

        self.volumetric_constant = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True)) 
                 

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
        rho[less_than_index] = self.stellar_parameters['rho_0']
        # If radius is greater than 1 solar radii...
        rho[else_index] = self.stellar_parameters['rho_0'] * torch.exp(1/self.stellar_parameters['h0']*(1/radius[else_index]-1)) #See equation 4 in Pascoe et al. 2019
        rho = torch.log(rho)

        # Simple temperature model (depends on radius)
        # If radius is less then 1 solar radii...
        temp[less_than_index] = self.t_photosphere
        # If radius is between 1 solar radii and R_s solar radii...
        R_s_index = torch.logical_and(radius > 1, radius <= self.stellar_parameters['Rs'])
        temp[R_s_index] = (radius[R_s_index]-1)*((self.stellar_parameters['T0']-self.t_photosphere)/(self.stellar_parameters['Rs'] - 1))+ self.t_photosphere #See equation 6 in Pascoe et al. 2019
        # If radius is greater than R_s solar radii, use constant...
        out_sun_index = radius > self.stellar_parameters['Rs']
        temp[out_sun_index]= self.stellar_parameters['T0']
        temp = torch.log10(temp)

        # Output density, temperature, absortpion and volumetric constant
        return {'rho_T': torch.stack((rho,temp), dim=-1), 'log_abs': self.log_absortpion , 'vol_c': self.volumetric_constant}


