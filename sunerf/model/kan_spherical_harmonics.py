import torch
from torch import nn
import sphericart.torch as sct
import numpy as np

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class SphericalBessel(nn.Module):
    def __init__(self, k_max: int = 1, l_max: int = 0):
        super().__init__()
        self.k_max = k_max
        self.l_max = l_max

    def forward(self, r):
        k = torch.linspace(1, self.k_max, self.k_max)
        r = r.to(torch.float64)[:,None]*k[None,:]
        y = torch.zeros([self.l_max]+[s for s in r.shape], dtype=torch.float64)

        lstart = self.l_max + int(torch.sqrt(torch.Tensor([10*self.l_max])))
        j2 = torch.zeros_like(r, dtype=torch.float64)
        j1 = torch.ones_like(r, dtype=torch.float64)

        for i in range(lstart, 0, -1):
            j0 = (2*i+1)/r * j1 - j2
            if i-1<self.l_max:
                y[i-1,...] = j0
            j2 = j1
            j1 = j0


        true_j0 = torch.sinc(r/torch.pi)
        y = y * true_j0/y[0,...]
        y = y.transpose(0,1)*torch.sqrt(torch.Tensor([2])/torch.pi)*k[None, None,:]
        return y.to(torch.float32)
    

class FourierSeries(nn.Module):
    def __init__(self, n_max: int = 1, scale = 1):
        super().__init__()
        self.n_max = n_max
        self.scale = scale

    def forward(self, t):
        n = torch.linspace(1, self.n_max, self.n_max)
        t = t.to(torch.float64)[:,None]
        
        return torch.cat((t*0+1, torch.sin(t*n[None,:]*2*torch.pi/self.scale), torch.cos(t*n[None,:]*2*torch.pi/self.scale)), dim=-1).to(torch.float32)
    

class SphericalHarmonicsModule(nn.Module):
    def __init__(self, l_max: int = 1):
        super().__init__()
        self.l_max = l_max
        self.sh = sct.SphericalHarmonics(l_max=8, normalized=True)

    def forward(self, xyz):
        sh_values = self.sh.compute(xyz.to(torch.float64))
        y = torch.zeros(sh_values.shape[0], self.l_max+1, 2*self.l_max+1)

        n = 0
        for l in np.arange(0, self.l_max+1):
            for m in np.arange(-l, l+1):
                y[:, l, m+self.l_max] = sh_values[:,n]
                n = n+1
        
        return y.to(torch.float32)
    

class OrthonormalTimeSphericalNeRF(nn.Module):
    def __init__(self,
                output_dim: int = 2,
                k_max: int = 1, 
                l_max: int = 1, 
                n_max: int = 1,
                t_scale: float = 1,
                spline_weight_init_scale: float = 0.1, 
                base_log_temperature: float = 5.0,
                base_log_density: float = 10.0):
        
        super().__init__()

        self.base_log_temperature = base_log_temperature
        self.base_log_density = base_log_density

        self.k_max = k_max
        self.l_max = l_max
        self.n_max = n_max
        self.t_scale = t_scale

        self.r_bessel = SphericalBessel(k_max=k_max, l_max=l_max+1)
        self.sh = SphericalHarmonicsModule(l_max=l_max)
        self.t_fourier = FourierSeries(n_max=n_max, scale=t_scale)

        self.spline_linear = SplineLinear((2*n_max+1)*k_max*(l_max+1)*(2*l_max+1), output_dim, spline_weight_init_scale)

        # Absorption for AIA, referred to instrument 0, EUVI-A refers to instrument 1, EUVI-B refers to instrument 2
        self.log_absortpion = nn.ParameterDict([
                                ['094',  torch.tensor(10.0, dtype=torch.float32)],
                                ['0131', torch.tensor(15.0, dtype=torch.float32)],
                                ['0171', torch.tensor(15.0, dtype=torch.float32)],
                                ['0193', torch.tensor(15.0, dtype=torch.float32)],
                                ['0211', torch.tensor(15.0, dtype=torch.float32)],
                                ['0304', torch.tensor(15.0, dtype=torch.float32)],
                                ['0335', torch.tensor(15.0, dtype=torch.float32)],
                                ['1171', torch.tensor(15.0, dtype=torch.float32)],
                                ['1195', torch.tensor(15.0, dtype=torch.float32)],
                                ['1284', torch.tensor(15.0, dtype=torch.float32)],
                                ['1304', torch.tensor(15.0, dtype=torch.float32)],
                                ['2171', torch.tensor(15.0, dtype=torch.float32)],
                                ['2195', torch.tensor(15.0, dtype=torch.float32)],
                                ['2284', torch.tensor(15.0, dtype=torch.float32)],
                                ['2304', torch.tensor(15.0, dtype=torch.float32)],
                        ])        

        self.volumetric_constant = nn.ParameterDict([
                                ['0', torch.tensor(1.0, dtype=torch.float32)],
                                ['1', torch.tensor(1.0, dtype=torch.float32)],
                                ['2', torch.tensor(1.0, dtype=torch.float32)],
                        ])

    def forward(self, x):
        fourier = self.t_fourier(x[:, 3])
        bessel = self.r_bessel(torch.sqrt(x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] + x[:, 2]*x[:, 2]))
        sh = self.sh(x[:, 0:3].contiguous())

        x = self.spline_linear((sh[:,:,:,None, None]*bessel[:,:,None,:, None]*fourier[:, None, None, None,:]).reshape(x.shape[0], -1))
        
        # Add base density
        x[:, 0] = x[:, 0] + self.base_log_density
        # Add base temperature
        x[:, 1] = x[:, 1] + self.base_log_temperature

        return {'inferences': x, 'log_abs': self.log_absortpion , 'vol_c': self.volumetric_constant}