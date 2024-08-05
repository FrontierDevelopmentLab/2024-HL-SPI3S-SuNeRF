from typing import Tuple, Optional, Callable
import torch
from torch import nn
import functools
import astropy.units as u
import numpy as np

class NeRF(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(
            self,
            d_input: int = 4,
            d_output: int = 2,
            n_layers: int = 8,
            d_filter: int = 256,
            skip: Tuple[int] = (4,),
            d_viewdirs: Optional[int] = None
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = Sine()
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
                 else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, d_output)

    def forward(
            self,
            x: torch.Tensor,
            viewdirs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x



class Sine(nn.Module):
    def __init__(self, w0: float = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, scale_factor: float = 2., log_space: bool = False, device=None):
        """

        Parameters
        ----------
        d_input: number of input dimensions
        n_freqs: number of frequencies used for encoding
        scale_factor: factor to adjust box size limit of 2pi (default 2; 4pi)
        log_space: use frequencies in powers of 2
        """
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [self.lambda_func]

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device        

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)
            print('freq bands', freq_bands)

        # Alternate sin and cos in model
        for freq in freq_bands:
            self.embed_fns.append(functools.partial(self.sin_encode, freq * scale_factor))
            self.embed_fns.append(functools.partial(self.cos_encode, freq * scale_factor))

    def lambda_func(self,x):
        return x.to(self.device)

    def sin_encode(self, scale, x):
            return torch.sin(2. * torch.pi * x.to(self.device) * scale.to(self.device)) 

    def cos_encode(self, scale, x):
            return torch.cos(2. * torch.pi * x.to(self.device) * scale.to(self.device)) 

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class GaussianEncoder(nn.Module):
    r"""
    Sine-cosine gaussian encoder for input points.

    https://arxiv.org/abs/2006.10739
    """

    def __init__(self, d_input: int, n_freqs: int, scale_factor: float = 1., scales: torch.Tensor = None, device=None):
        """

        Parameters
        ----------
        d_input: number of input dimensions
        n_freqs: number of frequencies used for encoding
        scale_factor: factor to adjust box size limit of 2pi (default 2; 4pi)
        log_space: use frequencies in powers of 2
        """
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.d_output = d_input * 2 * self.n_freqs
        self.scale_factor = scale_factor
        self.scales = scales
        self.embed_fns = []

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if scales is None:
            scales = torch.randn((n_freqs,d_input))*scale_factor

        for row in scales:
            self.embed_fns.append(functools.partial(self.sin_encode, row))
            self.embed_fns.append(functools.partial(self.cos_encode, row))

    def sin_encode(self, scale, x):
            return torch.sin(2. * torch.pi * x.to(self.device) * scale.to(self.device)) 

    def cos_encode(self, scale, x):
            return torch.cos(2. * torch.pi * x.to(self.device) * scale.to(self.device)) 

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)




class BesselEncoder(nn.Module):
    r"""
    Sine-cosine gaussian encoder for input points.

    https://arxiv.org/abs/2006.10739
    """

    def __init__(self, d_input: int, n_freqs: int, scale_factor: float = 1., scales: torch.Tensor = None, device=None):
        """

        Parameters
        ----------
        d_input: number of input dimensions
        n_freqs: number of frequencies used for encoding
        scale_factor: factor to adjust box size limit of 2pi (default 2; 4pi)
        log_space: use frequencies in powers of 2
        """
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.d_output = d_input * 2 * self.n_freqs
        self.scale_factor = scale_factor
        self.scales = scales
        self.embed_fns = []

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if scales is None:
            scales = torch.randn((n_freqs,d_input))*scale_factor

        for row in scales:
            self.embed_fns.append(functools.partial(self.i0_encode, row))
            self.embed_fns.append(functools.partial(self.i1_encode, row))

    def i0_encode(self, scale, x):
            return torch.special.i0( x.to(self.device) * scale.to(self.device)) 

    def i1_encode(self, scale, x):
            return torch.special.i1( x.to(self.device) * scale.to(self.device)) 

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)



class NeRF_absortpion(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(
            self,
            d_input: int = 4,
            d_output: int = 2,
            absortpion_output: int = 1,
            n_layers: int = 8,
            d_filter: int = 256,
            skip: Tuple[int] = (4,)
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
                 else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # If no viewdirs, use simpler output
        self.output = nn.Linear(d_filter, d_output)
        self.abs_coeff = nn.Parameter(torch.rand(1, 1))

    def forward(
            self,
            x: torch.Tensor,
            viewdirs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Simple output
        x = torch.cat(self.output(x), self.abs_coeff, dim=-1)

        return x



class NeRF_dens_temp(nn.Module):
  r"""
  Neural radiance fields module.
  """
  def __init__(
    self,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    d_input: int = 4,
    d_output: int = 2,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (4,),
    d_viewdirs: Optional[int] = None
  ):
    super().__init__()
    self.encoding_fn = encoding_fn
    self.d_input = d_input
    self.skip = skip
    self.act = Sine()
    self.d_viewdirs = d_viewdirs

    # Create model layers
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck layers
    if self.d_viewdirs is not None:
      # If using viewdirs, split alpha and RGB
      self.alpha_out = nn.Linear(d_filter, 1)
      self.rgb_filters = nn.Linear(d_filter, d_filter)
      self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      # If no viewdirs, use simpler output
      self.output = nn.Linear(d_filter, d_output)

    self.log_absortpion = nn.ParameterDict([
                                        ['94',  torch.rand(1, dtype=torch.float32)[0]],
                                        ['131', torch.rand(1, dtype=torch.float32)[0]],
                                        ['171', torch.rand(1, dtype=torch.float32)[0]],
                                        ['193', torch.rand(1, dtype=torch.float32)[0]],
                                        ['211', torch.rand(1, dtype=torch.float32)[0]],
                                        ['304', torch.rand(1, dtype=torch.float32)[0]],
                                        ['335', torch.rand(1, dtype=torch.float32)[0]]
                                ])

    self.volumetric_constant = nn.Parameter(torch.tensor(20.0, dtype=torch.float32, requires_grad=True))

  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    r"""
    Forward pass with optional view direction.
    """
    # Cannot use viewdirs if instantiated with d_viewdirs = None
    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

    # Apply forward pass up to bottleneck
    x_input = x
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # Apply bottleneck
    if self.d_viewdirs is not None:
      # Split alpha from network output
      alpha = self.alpha_out(x)

      # Pass through bottleneck to get RGB
      x = self.rgb_filters(x)
      x = torch.concat([x, viewdirs], dim=-1)
      x = self.act(self.branch(x))
      x = self.output(x)

      # Concatenate alphas to output
      x = torch.concat([x, alpha], dim=-1)
    else:
      # Simple output
      x = self.output(x)
    return x, self.log_absortpion, self.volumetric_constant


class dens_temp_ss(nn.Module):
  r"""
    basic atmospheric model 

    See (Pascoe et al. 2019) https://iopscience.iop.org/article/10.3847/1538-4357/ab3e39
  """
  def __init__(
    self,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    rho_0: Optional[float] = 8.477,   # np.log10(3.0e8/u.cm**3)
    base_rho: Optional[float] = 3.0,
    h0: Optional[float] = (60*u.Mm).to(u.solRad).value, 
    t0: Optional[float] = 6.14, #   np.log10(1.4e6*u.K) 
    base_temp: Optional[float] = 5.5
  ):
    super().__init__()

    self.encoding_fn = encoding_fn
    self.log_absortpion = nn.ParameterDict([
                                        ['94',  torch.tensor(4.4, dtype=torch.float32)],
                                        ['131', torch.tensor(4.2, dtype=torch.float32)],
                                        ['171', torch.tensor(4.0, dtype=torch.float32)],
                                        ['193', torch.tensor(3.8, dtype=torch.float32)],
                                        ['211', torch.tensor(3.6, dtype=torch.float32)],
                                        ['304', torch.tensor(3.4, dtype=torch.float32)],
                                        ['335', torch.tensor(3.2, dtype=torch.float32)]
                                ])

    self.volumetric_constant = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))

    # Simple star density parameters
    self.rho_0 = nn.Parameter(torch.tensor(rho_0, dtype=torch.float32, requires_grad=True))  # Learnable density
    # self.rho_0 = torch.tensor(rho_0)  # Learnable density
    self.base_rho = base_rho
    # self.h0 =  torch.tensor(h0)  # Learnable density scale height
    self.h0 = nn.Parameter(torch.tensor(h0, dtype=torch.float32, requires_grad=True))  # Learnable density scale height

    # Simple star temperature parameters
    self.t0 = nn.Parameter(torch.tensor(t0, dtype=torch.float32, requires_grad=True))  # Learnable constant temperature
    # self.t0 =  torch.tensor(t0)  # Learnable constant temperature
    self.base_temp = base_temp

  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    r"""
    Forward pass with optional view direction.

    basic atmospheric model 

    See (Pascoe et al. 2019) https://iopscience.iop.org/article/10.3847/1538-4357/ab3e39
    """

    # Simple star:

    # Radius (distance from the center of the sphere) in solar radii
    radius = torch.sqrt(x[:,0:3].pow(2).sum(-1))
    # Initialize density and temperature
    temp = torch.ones_like(radius)*nn.functional.relu(self.t0 -  self.base_temp)
    rho = torch.zeros_like(radius)

    #find indices of radii less than solRad and greater than solRad
    less_than_index = radius <= 1.0
    else_index = radius > 1.0

    # If radius is less then 1 solar radii...
    rho[less_than_index] = nn.functional.relu(self.rho_0 - self.base_rho)
    # If radius is greater than 1 solar radii...
    rho[else_index] = nn.functional.relu(self.rho_0 - self.base_rho) + (1.0/(nn.functional.relu(self.h0) + 1e-1)*(1.0/radius[else_index]-1)) * 0.4342944819032518
    ss_rho_temp = torch.stack((rho,temp), dim=-1)

    return ss_rho_temp, self.log_absortpion, self.volumetric_constant



class NeRF_dens_temp_ss(nn.Module):
  r"""
  Neural radiance fields module for density and temperature with the nerf working on top
  of a simple start model.
  """
  def __init__(
    self,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    d_input: int = 4,
    d_output: int = 2,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (4,),
    d_viewdirs: Optional[int] = None,
    rho_0: Optional[float] = 8.477,   # np.log10(3.0e8/u.cm**3)
    base_rho: Optional[float] = 3.0,
    h0: Optional[float] = (60*u.Mm).to(u.solRad).value, 
    t0: Optional[float] = 6.14, #   np.log10(1.4e6*u.K) 
    base_temp: Optional[float] = 5.5
  ):
    super().__init__()

    # NeRF
    self.encoding_fn = encoding_fn
    self.d_input = d_input
    self.skip = skip
    self.act = Sine()
    self.d_viewdirs = d_viewdirs

    # Create model layers
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck layers
    if self.d_viewdirs is not None:
      # If using viewdirs, split alpha and RGB
      self.alpha_out = nn.Linear(d_filter, 1)
      self.rgb_filters = nn.Linear(d_filter, d_filter)
      self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      # If no viewdirs, use simpler output
      self.output = nn.Linear(d_filter, d_output)

    self.log_absortpion = nn.ParameterDict([
                                        ['94',  torch.tensor(4.4, dtype=torch.float32)],
                                        ['131', torch.tensor(4.2, dtype=torch.float32)],
                                        ['171', torch.tensor(4.0, dtype=torch.float32)],
                                        ['193', torch.tensor(3.8, dtype=torch.float32)],
                                        ['211', torch.tensor(3.6, dtype=torch.float32)],
                                        ['304', torch.tensor(3.4, dtype=torch.float32)],
                                        ['335', torch.tensor(3.2, dtype=torch.float32)]
                                ])

    self.volumetric_constant = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))

    # Simple star density parameters
    self.rho_0 = nn.Parameter(torch.tensor(rho_0, dtype=torch.float32, requires_grad=True))  # Learnable density
    self.base_rho = base_rho
    self.h0 = nn.Parameter(torch.tensor(h0, dtype=torch.float32, requires_grad=True))  # Learnable density scale height

    # Simple star temperature parameters
    self.t0 = nn.Parameter(torch.tensor(t0, dtype=torch.float32, requires_grad=True))  # Learnable constant temperature
    self.base_temp = base_temp

  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    r"""
    Forward pass with optional view direction.

    includes a basic atmospheric model 

    See (Pascoe et al. 2019) https://iopscience.iop.org/article/10.3847/1538-4357/ab3e39
    """

    # Simple star:

    # Radius (distance from the center of the sphere) in solar radii
    radius = torch.sqrt(x[:,0:3].pow(2).sum(-1))
    # Initialize density and temperature
    temp = torch.ones_like(radius)*nn.functional.relu(self.t0 -  self.base_temp)
    rho = torch.zeros_like(radius)

    #find indices of radii less than solRad and greater than solRad
    less_than_index = radius <= 1.0
    else_index = radius > 1.0

    # If radius is less then 1 solar radii...
    rho[less_than_index] = nn.functional.relu(self.rho_0 - self.base_rho)
    # If radius is greater than 1 solar radii...
    rho[else_index] = nn.functional.relu(self.rho_0 - self.base_rho) + (1.0/(nn.functional.relu(self.h0) + 1e-10)*(1.0/radius[else_index]-1)) * 0.4342944819032518
    ss_rho_temp = torch.stack((rho,temp), dim=-1)


    # Cannot use viewdirs if instantiated with d_viewdirs = None
    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

    # Apply forward pass up to bottleneck
    x_input = x
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # Apply bottleneck
    if self.d_viewdirs is not None:
      # Split alpha from network output
      alpha = self.alpha_out(x)

      # Pass through bottleneck to get RGB
      x = self.rgb_filters(x)
      x = torch.concat([x, viewdirs], dim=-1)
      x = self.act(self.branch(x))
      x = self.output(x)

      # Concatenate alphas to output
      x = torch.concat([x, alpha], dim=-1)
    else:
      # Simple output
      x = self.output(x)

    x = x + ss_rho_temp

    return x, self.log_absortpion, self.volumetric_constant




class NeRF_dens_temp_ss_pw(nn.Module):
  r"""
  Neural radiance fields module for density and temperature with the nerf working on top
  of a simple start model.
  """
  def __init__(
    self,
    encoding_fn: Callable[[torch.Tensor], torch.Tensor],
    d_input: int = 4,
    d_output: int = 2,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (4,),
    d_viewdirs: Optional[int] = None,
    rho_0: Optional[float] = 3.0e8/u.cm**3,
    base_rho: Optional[float] = 10000.0,
    norm_rho: Optional[float] = 171.0,
    pow_rho: Optional[float] = 0.25,
    h0: Optional[float] = (60*u.Mm).to(u.solRad).value, 
    t0: Optional[float] = 6.14, #   np.log10(1.4e6*u.K) 
    base_temp: Optional[float] = 5.5
  ):
    super().__init__()

    # NeRF
    self.encoding_fn = encoding_fn
    self.d_input = d_input
    self.skip = skip
    self.act = Sine()
    self.d_viewdirs = d_viewdirs

    # Create model layers
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck layers
    if self.d_viewdirs is not None:
      # If using viewdirs, split alpha and RGB
      self.alpha_out = nn.Linear(d_filter, 1)
      self.rgb_filters = nn.Linear(d_filter, d_filter)
      self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      # If no viewdirs, use simpler output
      self.output = nn.Linear(d_filter, d_output)

    self.log_absortpion = nn.ParameterDict([
                                        ['94',  torch.tensor(4.4, dtype=torch.float32)],
                                        ['131', torch.tensor(4.2, dtype=torch.float32)],
                                        ['171', torch.tensor(4.0, dtype=torch.float32)],
                                        ['193', torch.tensor(3.8, dtype=torch.float32)],
                                        ['211', torch.tensor(3.6, dtype=torch.float32)],
                                        ['304', torch.tensor(3.4, dtype=torch.float32)],
                                        ['335', torch.tensor(3.2, dtype=torch.float32)]
                                ])

    # self.volumetric_constant = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
    self.volumetric_constant = 1.0

    # Simple star density parameters
    self.rho_0 = nn.Parameter(torch.tensor(np.power(rho_0, pow_rho)/norm_rho, dtype=torch.float32, requires_grad=True))  # Learnable density
    self.base_rho = base_rho
    self.pow_rho = pow_rho
    self.norm_rho = norm_rho
    self.h0 = nn.Parameter(torch.tensor(h0, dtype=torch.float32, requires_grad=True))  # Learnable density scale height

    # Simple star temperature parameters
    self.t0 = nn.Parameter(torch.tensor(t0, dtype=torch.float32, requires_grad=True))  # Learnable constant temperature
    self.base_temp = base_temp

  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    r"""
    Forward pass with optional view direction.

    includes a basic atmospheric model 

    See (Pascoe et al. 2019) https://iopscience.iop.org/article/10.3847/1538-4357/ab3e39
    """

    # Simple star:

    # Radius (distance from the center of the sphere) in solar radii
    radius = torch.sqrt(x[:,0:3].pow(2).sum(-1))
    # Initialize density and temperature
    temp = torch.ones_like(radius)*nn.functional.relu(self.t0 - self.base_temp)
    rho = torch.ones_like(radius)

    #find indices of radii less than solRad and greater than solRad\
    else_index = radius > 1.0

    # If radius is less then 1 solar radii...
    rho = rho*(nn.functional.relu(self.rho_0)*self.norm_rho).pow(1/self.pow_rho)
    # If radius is greater than 1 solar radii...
    rho[else_index] = rho[else_index] * torch.exp(1.0/(nn.functional.relu(self.h0) + 1e-10) * (1.0/radius[else_index]-1)) # See equation 4 in Pascoe et al. 2019
    rho = nn.functional.relu(rho - self.base_rho).pow(self.pow_rho)/self.norm_rho

    ss_rho_temp = torch.stack((rho,temp), dim=-1)


    # Cannot use viewdirs if instantiated with d_viewdirs = None
    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

    # Apply forward pass up to bottleneck
    x_input = x
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # Apply bottleneck
    if self.d_viewdirs is not None:
      # Split alpha from network output
      alpha = self.alpha_out(x)

      # Pass through bottleneck to get RGB
      x = self.rgb_filters(x)
      x = torch.concat([x, viewdirs], dim=-1)
      x = self.act(self.branch(x))
      x = self.output(x)

      # Concatenate alphas to output
      x = torch.concat([x, alpha], dim=-1)
    else:
      # Simple output
      x = self.output(x)

    x = x + ss_rho_temp

    return x, self.log_absortpion, self.volumetric_constant


def init_models(
        d_output: int,
        n_layers: int,
        d_filter: int,
        skip: Tuple[int],
        encoding: bool,
        encoder_kwargs: dict,
        use_single_model: Optional[bool] = False
):
    r"""_summary_

    Initialize models, and encoders for NeRF training.

    Args:
        d_input (int): Number of input dimensions (x,y,z,t)
        d_output (int): wavelength absorption + emission
        n_freqs (int): Number of encoding functions for samples
        n_layers (int): Number of layers in network bottleneck
        d_filter (int): Dimensions of linear layer filters
        log_space (bool): If set, frequencies scale in log space
        skip (Tuple[int]): Layers at which to apply input residual
    Returns:
        coarse_model:
        fine_model:
        encode:
    """

    # Encoders
    if encoding.lower() == 'gaussian':
        encoder = GaussianEncoder(**encoder_kwargs)
    elif encoding.lower() == 'bessel':
        encoder = BesselEncoder(**encoder_kwargs)
    else:
        encoder = PositionalEncoder(**encoder_kwargs)

    fine_model = NeRF_dens_temp_ss_pw(
        encoder.forward,
        encoder.d_output,
        d_output,
        n_layers=n_layers,
        d_filter=d_filter,
        skip=skip)

    # fine_model = dens_temp_ss(
    #     encoder.forward)

    model_params = list(fine_model.parameters())

    ss_params = []
    for name, param in fine_model.named_parameters():
      if param.requires_grad and 'output' not in name and 'layers' not in name:
          ss_params.append(param)

    nerf_params = []
    for name, param in fine_model.named_parameters():
      if param.requires_grad and ('output' in name or 'layers' in name or 'volumetric' in name or 'log_absortpion' in name):
          nerf_params.append(param)              

    if not use_single_model:
      coarse_model = NeRF_dens_temp_ss_pw(
          encoder.forward,
          encoder.d_output,
          d_output,
          n_layers=n_layers,
          d_filter=d_filter,
          skip=skip)

      # coarse_model = dens_temp_ss(
      #     encoder.forward)

      model_params = model_params + list(coarse_model.parameters())

      # for name, param in coarse_model.named_parameters():
      #   if param.requires_grad and 'output' not in name and 'layers' not in name:
      #       ss_params.append(param)
      #       print(name)

      # for name, param in coarse_model.named_parameters():
      #   if param.requires_grad and ('output' in name or 'layers' in name or 'volumetric' in name or 'log_absortpion' in name):
      #       nerf_params.append(param)
      #       print(name)               

    else:
        coarse_model = None

    return coarse_model, fine_model, model_params, ss_params, nerf_params
