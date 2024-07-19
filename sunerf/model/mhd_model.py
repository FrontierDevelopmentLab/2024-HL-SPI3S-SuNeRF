import torch
from torch import nn
import astropy.units as u
import glob
import os
from sunerf.data.mhd import rdhdf_3d
from scipy.interpolate import RegularGridInterpolator as rgi

class MHDModel(nn.Module):
    r""" Interpolation of MHD model such that it behaves like a trained NeRF
    """
    def __init__(self, data_path):
        """_summary_

        Parameters
        ----------
        data_path : str
        """            
        super().__init__()
        self.data_path = data_path
        self.density_files = sorted(glob.glob(os.path.join(data_path, 'rho', '*.h5')))
        self.temperature_files = sorted(glob.glob(os.path.join(data_path, 't', '*.h5')))
        self.ffirst = int(self.density_files[0].split('00')[1].split('.h5')[0])  #rho002531.h5
        self.flast = int(self.density_files[-1].split('00')[1].split('.h5')[0])

                            
        self.log_absortpion = nn.ParameterDict([
                                ['94',  torch.tensor(20.4, dtype=torch.float32)],
                                ['131', torch.tensor(20.2, dtype=torch.float32)],
                                ['171', torch.tensor(20.0, dtype=torch.float32)],
                                ['193', torch.tensor(19.8, dtype=torch.float32)],
                                ['211', torch.tensor(19.6, dtype=torch.float32)],
                                ['304', torch.tensor(19.4, dtype=torch.float32)],
                                ['335', torch.tensor(19.2, dtype=torch.float32)]
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

        # user-chosen coordinates (x,y,z) and time (t) for density and temperature to query
        x = query_points[:,0]
        y = query_points[:,1]
        z = query_points[:,2]
        t = query_points[:,3]

        output_density = torch.zeros_like(x)
        output_temperature = torch.zeros_like(x)

        # loop over only unique times
        for time in torch.uniqe(t):
            f1 = time * (self.flast - self.ffirst) + self.ffirst
            frame_fraction = f1 - int(f1)
            f2 = torch.ceil(f1).astype(torch.int)
            f1 = torch.floor(f1).astype(torch.int)

            # open files and read x, y, z, f for rho, t
            x1_rho, y1_rho, z1_rho, rho1 = rdhdf_3d(os.path.join(self.data_path, 'rho', f'rho00{f1}.h5'))
            x1_t, y1_t, z1_t, t1 = rdhdf_3d(os.path.join(self.data_path, 'rho', f't00{f1}.h5'))
            x2_rho, y2_rho, z2_rho, rho2 = rdhdf_3d(os.path.join(self.data_path, 'rho', f'rho00{f2}.h5'))
            x2_t, y2_t, z2_t, t2 = rdhdf_3d(os.path.join(self.data_path, 'rho', f't00{f2}.h5'))

            # create interpolating function in space for each file for rho, t
            f1_rho_interp = rgi((x1_rho, y1_rho, z1_rho), rho1)
            f1_t_interp = rgi((x1_t, y1_t, z1_t), t1)
            f2_rho_interp = rgi((x2_rho, y2_rho, z2_rho), rho2)
            f2_t_interp = rgi((x2_t, y2_t, z2_t), t2)

            # define logical mask to make interpolation
            mask = (t == time)
            x_mask = x[mask]
            y_mask = y[mask]
            z_mask = z[mask]

            # apply mask
            # TODO: reshape out of tensor
            f1_rho = f1_rho_interp((x_mask, y_mask, z_mask))
            f1_t = f1_t_interp((x_mask, y_mask, z_mask))
            f2_rho = f2_rho_interp((x_mask, y_mask, z_mask))
            f2_t = f2_t_interp((x_mask, y_mask, z_mask))
            
            output_density[mask] = (1-frame_fraction)*f1_rho + frame_fraction*f2_rho
            output_temperature[mask] = (1-frame_fraction)*f1_t + frame_fraction*f2_t



        # Output density, temperature, absortpion and volumetric constant
        return {'rho_T': torch.stack((output_density, output_temperature), dim=-1), 'log_abs': self.log_absortpion , 'vol_c': self.volumetric_constant}


