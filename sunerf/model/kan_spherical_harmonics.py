import torch
import torch.nn as nn
import torch.nn.functional as F
import sphericart.torch as sct
import numpy as np

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y

class SphericalBessel(nn.Module):
    def __init__(self, k_max: int = 1, l_max: int = 0):
        super().__init__()
        self.k_max = k_max
        self.l_max = l_max

    def forward(self, r):
        k = torch.linspace(1, self.k_max, self.k_max).to(r.device)
        r = r.to(torch.float64)[:,None]*k[None,:]
        y = torch.zeros([self.l_max]+[s for s in r.shape], dtype=torch.float64).to(r.device)

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
        y = y.transpose(0,1)*torch.sqrt(torch.Tensor([2]).to(r.device)/torch.pi)*k[None, None,:]
        return y.to(torch.float32)
    

class FourierSeries(nn.Module):
    def __init__(self, n_max: int = 1, scale = 1):
        super().__init__()
        self.n_max = n_max
        self.scale = scale

    def forward(self, t):
        n = torch.linspace(1, self.n_max, self.n_max).to(t.device)
        t = t[:,None]
        
        return torch.cat((t*0+1, torch.sin(t*n[None,:]*2*torch.pi/self.scale), torch.cos(t*n[None,:]*2*torch.pi/self.scale)), dim=-1).to(torch.float32)
    

class SphericalHarmonicsModule(nn.Module):
    def __init__(self, l_max: int = 1):
        super().__init__()
        self.l_max = l_max
        self.sh = sct.SphericalHarmonics(l_max=self.l_max, normalized=True)

    def forward(self, xyz):
        sh_values = self.sh.compute(xyz)
        y = torch.zeros(sh_values.shape[0], self.l_max+1, 2*self.l_max+1).to(xyz.device)

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
                                ['1193', torch.tensor(15.0, dtype=torch.float32)],
                                ['1211', torch.tensor(15.0, dtype=torch.float32)],
                                ['1304', torch.tensor(15.0, dtype=torch.float32)],
                                ['2171', torch.tensor(15.0, dtype=torch.float32)],
                                ['2193', torch.tensor(15.0, dtype=torch.float32)],
                                ['2211', torch.tensor(15.0, dtype=torch.float32)],
                                ['2304', torch.tensor(15.0, dtype=torch.float32)],
                        ])        

        self.volumetric_constant = nn.ParameterDict([
                                ['0', torch.tensor(1.0, dtype=torch.float32)],
                                ['1', torch.tensor(1.0, dtype=torch.float32)],
                                ['2', torch.tensor(1.0, dtype=torch.float32)],
                        ])

    def forward(self, x):
        fourier = self.t_fourier(x[:, 3])
        fourier[torch.isnan(fourier)] = 0
        bessel = self.r_bessel(torch.sqrt(x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] + x[:, 2]*x[:, 2]))
        bessel[torch.isnan(bessel)] = 0
        sh = self.sh(x[:, 0:3].contiguous())
        sh[torch.isnan(sh)] = 0

        x = self.spline_linear((sh[:,:,:,None, None]*bessel[:,:,None,:, None]*fourier[:, None, None, None,:]).reshape(x.shape[0], -1))
        
        # Add base density
        x[:, 0] = x[:, 0] + self.base_log_density
        # Add base temperature
        x[:, 1] = x[:, 1] + self.base_log_temperature

        if x.isnan().any():
            print('nan')

        return {'inferences': x, 'log_abs': self.log_absortpion , 'vol_c': self.volumetric_constant}