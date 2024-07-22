import torch
from torch import nn
import glob
import os
from sunerf.data.mhd.psi_io import rdhdf_3d
from scipy.interpolate import RegularGridInterpolator as rgi
import numpy as np

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

        # Convert to spherical coordinates
        r = torch.sqrt(x**2 + y**2 + z**2)

        th = torch.arccos(z/r)
        phi = torch.arctan2(y,x)
        phi[phi<0] += 2*np.pi

        output_density = torch.zeros_like(x)
        output_temperature = torch.zeros_like(x)

        # loop over only unique times
        for time in torch.unique(t):
            f1 = time * (self.flast - self.ffirst) + self.ffirst
            frame_fraction = f1 - int(f1)
            f2 = torch.ceil(f1).type(torch.int)
            f1 = torch.floor(f1).type(torch.int)

            # open files; read x_mhd, y_mhd, z_mhd, f; and interpolate for rho, t
            r_mhd, th_mhd, phi_mhd, rho1 = rdhdf_3d(os.path.join(self.data_path, 'rho', f'rho00{f1}.h5'))
            f1_rho_interp = rgi((phi_mhd, th_mhd, r_mhd), rho1, bounds_error=False, fill_value=1e-10)

            r_mhd, th_mhd, phi_mhd, t1 = rdhdf_3d(os.path.join(self.data_path, 't', f't00{f1}.h5'))
            f1_t_interp = rgi((phi_mhd, th_mhd, r_mhd), t1, bounds_error=False, fill_value=1e-10)

            r_mhd, th_mhd, phi_mhd, rho2 = rdhdf_3d(os.path.join(self.data_path, 'rho', f'rho00{f2}.h5'))
            f2_rho_interp = rgi((phi_mhd, th_mhd, r_mhd), rho2, bounds_error=False, fill_value=1e-10)

            r_mhd, th_mhd, phi_mhd, t2 = rdhdf_3d(os.path.join(self.data_path, 't', f't00{f2}.h5'))
            f2_t_interp = rgi((phi_mhd, th_mhd, r_mhd), t2, bounds_error=False, fill_value=1e-10)

            # define logical mask to make interpolation
            mask = (t == time).cpu().numpy()
            r_mask = r[mask].cpu().numpy()
            th_mask = th[mask].cpu().numpy()
            phi_mask = phi[mask].cpu().numpy()

            # apply mask
            f1_rho = torch.Tensor(f1_rho_interp((phi_mask, th_mask, r_mask))).to(t.device)
            f1_t = torch.Tensor(f1_t_interp((phi_mask, th_mask, r_mask))).to(t.device)
            f2_rho = torch.Tensor(f2_rho_interp((phi_mask, th_mask, r_mask))).to(t.device)
            f2_t = torch.Tensor(f2_t_interp((phi_mask, th_mask, r_mask))).to(t.device)
            
            output_density[mask] = torch.log((1-frame_fraction)*f1_rho + frame_fraction*f2_rho)
            output_temperature[mask] = torch.log10(1e6*((1-frame_fraction)*f1_t + frame_fraction*f2_t))

            # TODO: make mhd render (maybe make another class or file?)

        # Output density, temperature, absortpion and volumetric constant
        return {'rho_T': torch.stack((output_density, output_temperature), dim=-1), 'log_abs': self.log_absortpion , 'vol_c': self.volumetric_constant}


