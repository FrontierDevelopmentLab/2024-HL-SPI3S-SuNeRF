import torch
from torch import nn


class GenericModel(nn.Module):

    def __init__(self, in_dim, out_dim, dim=512, n_layers=8, encoding=None, activation='sine'):
        super().__init__()
        if encoding is None or encoding == 'none':
            self.d_in = nn.Linear(in_dim, dim)
        elif encoding == 'positional':
            posenc = PositionalEncoding(10, in_dim)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == 'gaussian':
            posenc = GaussianPositionalEncoding(20, in_dim)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            raise NotImplementedError(f'Unknown encoding {encoding}')
        lin = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, out_dim)
        activation_mapping = {'relu': nn.ReLU, 'swish': Swish, 'tanh': nn.Tanh, 'sine': Sine}
        activation_f = activation_mapping[activation]
        self.in_activation = activation_f()
        self.activations = nn.ModuleList([activation_f() for _ in range(n_layers)])

    def forward(self, x):
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        x = self.d_out(x)
        return x


class EmissionModel(GenericModel):

    def __init__(self, n_channels=1, encoding='positional', **kwargs):
        super().__init__(in_dim=4, out_dim=n_channels * 2, encoding=encoding, **kwargs)
        self.n_channels = n_channels

    def forward(self, x):
        out = super().forward(x)
        emission = torch.exp(out[..., :self.n_channels])
        alpha = nn.functional.relu(out[..., self.n_channels:])
        return {'emission': emission, 'alpha': alpha}

class PlasmaModel(GenericModel):

    def __init__(self, encoding='positional', **kwargs):
        super().__init__(in_dim=4, out_dim=2, encoding=encoding, **kwargs)

        self.T_range = nn.Parameter(torch.tensor([5.0, 8.0], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        out = super().forward(x)
        log_T = out[..., 0:1] * (self.T_range[1] - self.T_range[0]) + self.T_range[0]
        log_ne = out[..., 1:2]
        return {'log_T': log_T, 'log_ne': log_ne, 'T': 10 ** log_T, 'ne': 10 ** log_ne}


class AbsorptionModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(in_dim=2, out_dim=1, **kwargs)

    def forward(self, x):
        out = super().forward(x)
        return {'log_nu': out, 'nu': 10 ** out}


class Sine(nn.Module):
    def __init__(self, w0: float = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


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

    def __init__(self, num_freqs, in_features, max_freq=10):
        super().__init__()
        frequencies = 2 ** torch.linspace(-1, max_freq - 2, num_freqs)
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.d_output = in_features * (1 + num_freqs * 2)

    def forward(self, x):
        encoded = x[..., None] * self.frequencies.reshape([1] * len(x.shape) + [-1])
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded), x], -1)
        return encoded


class GaussianPositionalEncoding(nn.Module):

    def __init__(self, num_freqs, d_input, scale=1., log_scale=True):
        super().__init__()
        if log_scale:
            frequencies = 2 ** (torch.randn(num_freqs, d_input, dtype=torch.float32) * scale) * torch.pi
        else:
            frequencies = torch.randn(num_freqs, d_input, dtype=torch.float32) * scale * torch.pi
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.d_output = d_input * (1 + num_freqs * 2)

    def forward(self, x):
        encoded = x[..., None, :] * self.frequencies.reshape([1] * (len(x.shape) - 1) + [*self.frequencies.shape])
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded), x], -1)
        return encoded
