import torch
import torch.nn as nn
import torch.nn.functional as F
import sphericart.torch as sct
import numpy as np

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super(SplineLinear, self).__init__(in_features, out_features, bias=False, **kw)

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
        super(RadialBasisFunction, self).__init__()
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
        super(FastKANLayer, self).__init__()
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
    def __init__(self, l_max: int = 1):
        super().__init__()
        self.l_max = l_max

        self.l = None
        self.m = None
        for i in range(l_max+1):
            if i ==0:
                self.l = torch.Tensor([0])
                self.m = torch.Tensor([0])
            else:
                m_array = torch.arange(-i, i+1)
                self.l = torch.cat((self.l, m_array*0+i))
                self.m = torch.cat((self.m, m_array))        

    def bessel_down(self, r, k):

        r = r.to(torch.float64)[...,None]*k.to(torch.float64)
        y = torch.zeros_like(r, dtype=torch.float64)

        lstart = self.l_max + int(torch.sqrt(torch.Tensor([10*self.l_max])))
        j2 = torch.zeros_like(r, dtype=torch.float64)
        j1 = torch.ones_like(r, dtype=torch.float64)

        for i in range(lstart, 0, -1):
            j0 = (2*i+1)/r * j1 - j2
            if i-1<self.l_max+1:
                y[..., i-1==self.l] = j0[..., i-1==self.l]
            j2 = j1
            j1 = j0

        true_j0 = torch.sinc(r/torch.pi)
        y = y / j0
        y = y * true_j0

        y[torch.logical_and(r<1e-20, self.l>0)] = 0
        y[torch.logical_and(r<1e-20, self.l==0)] = 1

        y = y*torch.sqrt(torch.Tensor([2])/torch.pi)*k
        return y.to(torch.float32)
    

    def bessel_up(self, r, k):

        r = r.to(torch.float64)[...,None]*k.to(torch.float64)
        y = torch.zeros_like(r, dtype=torch.float64)

        j0 =  torch.sin(r)/r
        y[..., self.l==0] = j0[..., self.l==0]

        j1 = j0/r - torch.cos(r)/r
        y[..., self.l==1] = j1[..., self.l==1]

        for i in range(1, self.l_max):
            j2 = (2*i+1)/r*j1 - j0
            y[..., self.l==i+1] = j2[..., self.l==i+1]
            j0 = j1
            j1 = j2

        y[torch.abs(r)<1e-30] = 0

        y = y*torch.sqrt(torch.Tensor([2])/torch.pi)*k
        return y.to(torch.float32)


    def forward(self, r, k):
        bessel_up = self.bessel_up(r, k)
        bessel_dwn = self.bessel_down(r, k)
        r = r[...,None]*k

        bessel_up[..., r<self.l] = bessel_dwn[..., r<self.l]

        return bessel_up
    
class FourierModes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, scale, phase):
        t = 2*torch.pi*(t.to(torch.float64)[...,None]*scale.to(torch.float64) - phase.to(torch.float64))
        
        return torch.cos(t).to(torch.float32)    

class SphericalHarmonicsModule(nn.Module):
    def __init__(self, l_max: int = 1):
        super().__init__()
        self.l_max = l_max
        self.sh = sct.SphericalHarmonics(l_max=self.l_max)

    def forward(self, xyz):
        sh_values = self.sh.compute(xyz)        
        return sh_values
    

class OrthonormalTimeSphericalBesselNeRF(nn.Module):
    def __init__(self,
                output_dim: int = 2, 
                l_max: int = 1, 
                spline_weight_init_scale: float = 0.1, 
                base_log_temperature: float = 5.0,
                base_log_density: float = 10.0):
        
        super().__init__()

        self.base_log_temperature = base_log_temperature
        self.base_log_density = base_log_density
        self.spline_linear = SplineLinear(in_features=(l_max+1)*(l_max+1), out_features=output_dim, init_scale=spline_weight_init_scale)

        self.l_max = l_max

        self.r_bessel = SphericalBessel(l_max=l_max)
        self.sh = SphericalHarmonicsModule(l_max=l_max)
        self.t_fourier = FourierModes()

        self.radius_scale = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))
        self.time_scale = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))
        self.time_phase = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))


        # Absorption for AIA, referred to instrument 0, EUVI-A refers to instrument 1, EUVI-B refers to instrument 2
        self.log_absortpion = nn.Parameter(20.0*torch.tensor([[1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0]], dtype=torch.float32, requires_grad=True)) 

        # Tensor with volumetric constant for all instruments.
        #  Position 0 (AIA), position 1 (EUVIA), and position 2 (EUVB)
        self.volumetric_constant = nn.Parameter(torch.tensor([1., 1., 1.,], dtype=torch.float32, requires_grad=True)) 


    def forward(self, x):
        fourier = self.t_fourier(x[:, 3], self.time_scale, self.time_phase)
        bessel = self.r_bessel(torch.sqrt(x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] + x[:, 2]*x[:, 2]), self.radius_scale)
        sh = self.sh(x[:, 0:3].contiguous().to(torch.float64)).to(torch.float32)

        # x = self.spline_linear((sh[:,:,:,None, None]*bessel[:,:,None,:, None]*fourier[:, None, None, None,:]).reshape(x.shape[0], -1))
        x = torch.abs(self.spline_linear(fourier*bessel*sh))
        
        # Add base density
        x[:, 0] = x[:, 0] + self.base_log_density
        # Add base temperature
        x[:, 1] = x[:, 1] + self.base_log_temperature

        return {'RhoT': x, 'log_abs': self.log_absortpion , 'vol_c': self.volumetric_constant}
    
class OrthonormalTimeSphericalRFourierNeRF(nn.Module):
    def __init__(self,
                output_dim: int = 2, 
                l_max: int = 1, 
                spline_weight_init_scale: float = 1.0, 
                base_log_temperature: float = 5.0,
                base_log_density: float = 10.0):
        
        super().__init__()

        self.base_log_temperature = base_log_temperature
        self.base_log_density = base_log_density
        self.spline_linear = SplineLinear(in_features=2*(l_max+1)*(l_max+1), out_features=output_dim, init_scale=spline_weight_init_scale)

        self.l_max = l_max

        self.sh = SphericalHarmonicsModule(l_max=l_max)
        self.t_fourier = FourierModes()

        self.r_scale_no_t = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))
        self.r_phase_no_t = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))
        self.r_scale_t = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))
        self.r_phase_t = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))
        self.t_scale = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))
        self.t_phase = nn.Parameter(torch.ones((l_max+1)*(l_max+1), dtype=torch.float32, requires_grad=True))


        # Absorption for AIA, referred to instrument 0, EUVI-A refers to instrument 1, EUVI-B refers to instrument 2
        self.log_absortpion = nn.Parameter(20.0*torch.tensor([[1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0],
                                                         [1.0, 1.0, 1.0]], dtype=torch.float32, requires_grad=True)) 

        # Tensor with volumetric constant for all instruments.
        #  Position 0 (AIA), position 1 (EUVIA), and position 2 (EUVB)
        self.volumetric_constant = nn.Parameter(torch.tensor([1., 1., 1.,], dtype=torch.float32, requires_grad=True)) 


    def forward(self, x):
        fourier_r_no_t = self.t_fourier(torch.sqrt(x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] + x[:, 2]*x[:, 2]), self.r_scale_no_t, self.r_phase_no_t)/torch.sqrt(x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] + x[:, 2]*x[:, 2])[...,None]
        fourier_r_t = self.t_fourier(torch.sqrt(x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] + x[:, 2]*x[:, 2]), self.r_scale_t, self.r_phase_t)/torch.sqrt(x[:, 0]*x[:, 0] + x[:, 1]*x[:, 1] + x[:, 2]*x[:, 2])[...,None]
        fourier_t = self.t_fourier(x[:, 3], self.t_scale, self.t_phase)
        sh = self.sh(x[:, 0:3].contiguous().to(torch.float64)).to(torch.float32)

        # x = self.spline_linear((sh[:,:,:,None, None]*bessel[:,:,None,:, None]*fourier[:, None, None, None,:]).reshape(x.shape[0], -1))
        x = torch.abs(self.spline_linear(torch.cat((fourier_t*fourier_r_t*sh, fourier_r_no_t*sh), dim=-1)))
        
        # Add base density
        x[:, 0] = x[:, 0] + self.base_log_density
        # Add base temperature
        x[:, 1] = x[:, 1] + self.base_log_temperature

        return {'RhoT': x, 'log_abs': self.log_absortpion , 'vol_c': self.volumetric_constant}