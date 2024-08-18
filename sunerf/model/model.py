from typing import Tuple

import torch
from torch import nn


class NeRF(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(
            self,
            d_input: int = 4,
            d_output: int = 2,
            n_layers: int = 8,
            d_filter: int = 512,
            skip: Tuple[int] = (),
            encoding='positional'
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = Sine()

        # encoding_config = {'type': 'positional', 'num_freqs': 20} if encoding_config is None else encoding_config
        # encoding_type = encoding_config.pop('type')
        if encoding == 'positional':
            enc = PositionalEncoding(d_input=d_input, n_freqs=10)
            in_layer = nn.Linear(enc.d_output, d_filter)
            self.in_layer = nn.Sequential(enc, in_layer)
        else:
            self.in_layer = nn.Linear(d_input, d_filter)
        # elif encoding_type == 'none' or encoding_type is None:
        #     self.in_layer = nn.Linear(d_input, d_filter)
        # else:
        #     raise ValueError(f'Unknown encoding type {encoding_type}')

        # Create model layers
        self.layers = nn.ModuleList([nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)])

        self.out_layer = nn.Linear(d_filter, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """

        # Apply forward pass
        x = self.act(self.in_layer(x))
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            # if i in self.skip:
            #     x = torch.cat([x, x_input], dim=-1)
        x = self.out_layer(x)

        return {'inferences':x}


class EmissionModel(NeRF):

    def __init__(self, **kwargs):
        super().__init__(d_input=4, d_output=2, **kwargs)


class Sine(nn.Module):
    def __init__(self, w0: float = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class TrainablePositionalEncoding(nn.Module):

    def __init__(self, d_input, n_freqs=20):
        super().__init__()
        frequencies = torch.stack([torch.linspace(-3, 9, n_freqs, dtype=torch.float32) for _ in range(d_input)], -1)
        self.frequencies = nn.Parameter(frequencies[None, :, :], requires_grad=True)
        self.d_output = n_freqs * 2 * d_input

    def forward(self, x):
        # x = (batch, rays, coords)
        encoded = x[:, None, :] * torch.pi * 2 ** self.frequencies
        normalization = (torch.pi * 2 ** self.frequencies)
        encoded = torch.cat([torch.sin(encoded) / normalization, torch.cos(encoded) / normalization], -1)
        encoded = encoded.reshape(x.shape[0], -1)
        return encoded


class PositionalEncoding(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, scale_factor: float = 2., log_space: bool = True):
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

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)
            print('freq bands', freq_bands)

        self.register_buffer('freq_bands', freq_bands)
        self.scale_factor = scale_factor

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        f = self.freq_bands[None, :, None]
        enc = [x,
               (torch.sin(x[:, None, :] * f / self.scale_factor)).reshape(x.shape[0], -1),
               (torch.cos(x[:, None, :] * f / self.scale_factor)).reshape(x.shape[0], -1)
               ]
        return torch.concat(enc, dim=-1)



class NeRF_DT(NeRF):
    r"""
    Neural radiance fields module.
    """

    def __init__(
            self,
            d_input: int = 4,
            d_output: int = 2,
            n_layers: int = 8,
            d_filter: int = 512,
            skip: Tuple[int] = (),
            encoding='positional', 
            base_log_temperature: float = 5.0,
            base_log_density: float = 10.0,
    ):
        super().__init__(d_input=d_input, d_output=d_output, n_layers=n_layers, d_filter=d_filter, skip=skip, encoding=encoding)

        self.base_log_temperature = base_log_temperature
        self.base_log_density = base_log_density

        self.log_absortpion = nn.ParameterDict([
                                ['94',  torch.tensor(1.e-6, dtype=torch.float32)],
                                ['131', torch.tensor(1.e-6, dtype=torch.float32)],
                                ['171', torch.tensor(1.e-6, dtype=torch.float32)],
                                ['193', torch.tensor(1.e-6, dtype=torch.float32)],
                                ['211', torch.tensor(1.e-6, dtype=torch.float32)],
                                ['304', torch.tensor(1.e-6, dtype=torch.float32)],
                                ['335', torch.tensor(1.e-6, dtype=torch.float32)]
                        ])

        self.volumetric_constant = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """

        # Apply forward pass
        x = self.act(self.in_layer(x))
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            # if i in self.skip:
            #     x = torch.cat([x, x_input], dim=-1)
        x = self.out_layer(x)

        # Add base density
        x[:, 0] = x[:, 0] + self.base_log_density
        # Add base temperature
        x[:, 1] = x[:, 1] + self.base_log_temperature

        return {'inferences': x, 'log_abs': self.log_absortpion , 'vol_c': self.volumetric_constant}

# class NeRF_dens_temp(nn.Module):
#   r"""
#   Neural radiance fields module.
#   """
#   def __init__(
#     self,
#     encoding_fn: Callable[[torch.Tensor], torch.Tensor],
#     d_input: int = 4,
#     d_output: int = 2,
#     n_layers: int = 8,
#     d_filter: int = 256,
#     skip: Tuple[int] = (4,),
#     d_viewdirs: Optional[int] = None
#   ):
#     super().__init__()
#     self.encoding_fn = encoding_fn
#     self.d_input = d_input
#     self.skip = skip
#     self.act = Sine()
#     self.d_viewdirs = d_viewdirs

#     # Create model layers
#     self.layers = nn.ModuleList(
#       [nn.Linear(self.d_input, d_filter)] +
#       [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
#        else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
#     )

#     # Bottleneck layers
#     if self.d_viewdirs is not None:
#       # If using viewdirs, split alpha and RGB
#       self.alpha_out = nn.Linear(d_filter, 1)
#       self.rgb_filters = nn.Linear(d_filter, d_filter)
#       self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
#       self.output = nn.Linear(d_filter // 2, 3)
#     else:
#       # If no viewdirs, use simpler output
#       self.output = nn.Linear(d_filter, d_output)

#     self.log_absortpion = nn.ParameterDict([
#                                         ['94',  torch.rand(1, dtype=torch.float32)[0]],
#                                         ['131', torch.rand(1, dtype=torch.float32)[0]],
#                                         ['171', torch.rand(1, dtype=torch.float32)[0]],
#                                         ['193', torch.rand(1, dtype=torch.float32)[0]],
#                                         ['211', torch.rand(1, dtype=torch.float32)[0]],
#                                         ['304', torch.rand(1, dtype=torch.float32)[0]],
#                                         ['335', torch.rand(1, dtype=torch.float32)[0]]
#                                 ])

#     self.volumetric_constant = nn.Parameter(torch.tensor(20.0, dtype=torch.float32, requires_grad=True))

#   def forward(
#     self,
#     x: torch.Tensor,
#     viewdirs: Optional[torch.Tensor] = None,
#   ) -> torch.Tensor:
#     r"""
#     Forward pass with optional view direction.
#     """
#     # Cannot use viewdirs if instantiated with d_viewdirs = None
#     if self.d_viewdirs is None and viewdirs is not None:
#       raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

#     # Apply forward pass up to bottleneck
#     x_input = x
#     for i, layer in enumerate(self.layers):
#       x = self.act(layer(x))
#       if i in self.skip:
#         x = torch.cat([x, x_input], dim=-1)

#     # Apply bottleneck
#     if self.d_viewdirs is not None:
#       # Split alpha from network output
#       alpha = self.alpha_out(x)

#       # Pass through bottleneck to get RGB
#       x = self.rgb_filters(x)
#       x = torch.concat([x, viewdirs], dim=-1)
#       x = self.act(self.branch(x))
#       x = self.output(x)

#       # Concatenate alphas to output
#       x = torch.concat([x, alpha], dim=-1)
#     else:
#       # Simple output
#       x = self.output(x)
#     return x, self.log_absortpion, self.volumetric_constant