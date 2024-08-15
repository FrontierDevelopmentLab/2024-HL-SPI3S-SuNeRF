import torch
from torch import nn
import glob
import os
from sunerf.data.mhd.psi_io import rdhdf_3d
import cupy as cp
from scipy.interpolate import RegularGridInterpolator as rgi
from cupyx.scipy.interpolate import RegularGridInterpolator as rgi_gpu
import numpy as np

class MHDModel(nn.Module):
    r""" Interpolation of MHD model such that it behaves like a trained NeRF
    """
    def __init__(self, data_path, device=None):
        """_summary_

        Parameters
        ----------
        data_path : str
        """            
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device

        self.data_path = data_path
        self.density_files = sorted(glob.glob(os.path.join(data_path, 'rho', '*.h5')))
        self.temperature_files = sorted(glob.glob(os.path.join(data_path, 't', '*.h5')))
        self.ffirst = int(self.density_files[0].split('00')[1].split('.h5')[0])  # rho002531.h5
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

    def interp(self, phi, theta, r, f, var, method='linear', fill_value=1e-10):
        """Interpolation of MHD data

        Args:
            phi (float): azimuthal angle
            theta (float): polar angle
            r (float): radial distance
            f (int): frame number
            var (str): variable to interpolate
            method (str, optional): interpolation type. Defaults to 'linear'.
            fill_value (float, optional): fill value. Defaults to 1e-10.

        Returns:
            torch.Tensor: interpolated data
        """

        # Read data
        r_mhd, th_mhd, phi_mhd, data = rdhdf_3d(os.path.join(self.data_path, var, f'{var}00{f}.h5'))
        # Filter data
        data[np.where(data < 0)] = fill_value

        # Interpolation based on device
        if self.device.type == 'cuda1':
            f1_interp = rgi_gpu((cp.array(phi_mhd), cp.array(th_mhd), cp.array(r_mhd)), cp.array(data),
                                method=method, bounds_error=False, fill_value=fill_value)
            return cp.asnumpy(f1_interp((cp.array(phi), cp.array(theta), cp.array(r))))
        else:
            f1_interp = rgi((phi_mhd, th_mhd, r_mhd), data,
                            method=method, bounds_error=False, fill_value=fill_value)
            return f1_interp((phi, theta, r))

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
        x = query_points[:, 0]
        y = query_points[:, 1]
        z = query_points[:, 2]
        t = query_points[:, 3]

        # Convert to spherical coordinates
        r = torch.sqrt(x**2 + y**2 + z**2)
        th = torch.arccos(z/r)
        phi = torch.arctan2(y, x)
        phi[phi < 0] += 2*np.pi

        # Initialize output tensors filled with zeros with shape x
        output_density = torch.zeros_like(x)
        output_temperature = torch.zeros_like(x)
        fill_value = 1.e-10
        interp_type = 'linear'

        # loop over only unique times
        for time in torch.unique(t):

            # Define logical mask to make interpolation
            mask = (t == time).cpu().numpy()
            r_mask = r[mask].cpu().numpy()
            th_mask = th[mask].cpu().numpy()
            phi_mask = phi[mask].cpu().numpy()

            # Interpolation of frames
            f1 = time * (self.flast - self.ffirst) + self.ffirst
            frame_fraction = f1 - int(f1)
            f2 = torch.ceil(f1).type(torch.int)
            f1 = torch.floor(f1).type(torch.int)

            # Interpolation of density and temperature
            f1_rho = torch.Tensor(self.interp(phi_mask, th_mask, r_mask, f1, 'rho',
                                  method=interp_type, fill_value=fill_value)).to(t.device)
            f1_t = torch.Tensor(self.interp(phi_mask, th_mask, r_mask, f1, 't',
                                method=interp_type, fill_value=fill_value)).to(t.device)
            f2_rho = torch.Tensor(self.interp(phi_mask, th_mask, r_mask, f2, 'rho',
                                  method=interp_type, fill_value=fill_value)).to(t.device)
            f2_t = torch.Tensor(self.interp(phi_mask, th_mask, r_mask, f2, 't',
                                method=interp_type, fill_value=fill_value)).to(t.device)

            # Linear time interpolation of density and temperature
            output_density[mask] = torch.log((1-frame_fraction)*f1_rho + frame_fraction*f2_rho)
            output_temperature[mask] = torch.log10(1e6*((1-frame_fraction)*f1_t + frame_fraction*f2_t))

        # Output density, temperature, absorption and volumetric constant
        return {'inferences': torch.stack((output_density, output_temperature), dim=-1), 'log_abs': self.log_absortpion,
                'vol_c': self.volumetric_constant}
