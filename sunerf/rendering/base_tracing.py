import torch
from torch import nn
import datetime
from sunerf.model.sunerf_nerf_models import NeRF
from sunerf.model.stellar_model import SimpleStar
from sunerf.train.sampling import SphericalSampler, HierarchicalSampler, StratifiedSampler, NonUniformStratifiedSampler

class SuNeRFRendering(nn.Module):

    def __init__(self, Rs_per_ds, sampling_config=None, hierarchical_sampling_config=None, model=NeRF, model_config=None, use_fine_model=True): # TODO: adopt do be used for different models
        super(SuNeRFRendering, self).__init__()
        self.Rs_per_ds = Rs_per_ds
        # Rs_per_ds --> Solar Radii per distance

        # set default configurations if none provided
        hierarchical_sampling_config = {'type': 'hierarchical'} \
            if hierarchical_sampling_config is None else hierarchical_sampling_config
        sampling_config = {'type': 'stratified'} if sampling_config is None else sampling_config
        model_config = {} if model_config is None else model_config

        # setup sampling strategy based on configuration
        sampling_type = sampling_config.pop('type')
        if sampling_type == 'spherical':
            self.sampler = SphericalSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        elif sampling_type == 'stratified':
            self.sampler = StratifiedSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        elif sampling_type == 'non-uniform-stratified':
            self.sampler = NonUniformStratifiedSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {sampling_type}')

        # setup hierarchical sampling method
        hierarchical_sampling_type = hierarchical_sampling_config.pop('type')
        if hierarchical_sampling_type == 'hierarchical':
            self.sampler_hierarchical = HierarchicalSampler(**hierarchical_sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {hierarchical_sampling_type}')

        self.use_fine_model = use_fine_model
        # setup coarse and fine models
        self.coarse_model = model(**model_config)
        if use_fine_model:
            self.fine_model = model(**model_config)

    def regularization(self, distance, regularizing_quantity):
        return torch.relu(distance[:,:,None] - 1.2 / self.Rs_per_ds) * (1 - regularizing_quantity)

    def forward(self, rays_o, rays_d, times, wavelengths=None, instruments=None):
        r"""_summary_
        		Compute forward pass through model.

        		Args:
        			rays_o (tensor): Origin of rays
        			rays_d (tensor): Direction of rays
        			times (tensor): Times of maps
        		Returns:
        			outputs: Synthesized filtergrams/images.
        		"""

        # Sample query points along each ray using defined sampling method
        sampling_out = self.sampler(rays_o, rays_d)
        query_points, z_vals = sampling_out['points'], sampling_out['z_vals']
        # z_vals --> Distance along the ray

        # add time to query points
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)  # --> (x, y, z, t)

        # Coarse model pass. (Density and temperature at each sample points along the ray)
        if wavelengths is None:
            coarse_out = self._render(self.coarse_model, query_points_time, rays_d, rays_o, z_vals)
        else:
            coarse_out = self._render(self.coarse_model, query_points_time, rays_d, rays_o, z_vals, wavelengths, instruments)
        outputs = {'z_vals_stratified': z_vals, 'coarse_image': coarse_out['image']}      

        # Fine model pass.
        if self.use_fine_model:
            # Apply hierarchical sampling for fine query points.
            hierarchical_out = self.sampler_hierarchical(rays_o, rays_d, z_vals, coarse_out['weights'])
            query_points, z_vals_combined, z_hierarch = (hierarchical_out['points'],
                                                        hierarchical_out['z_vals'],
                                                        hierarchical_out['new_z_samples'])


            # add time to query points = expand to dimensions of query points and slice one dimension
            exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
            query_points_time = torch.cat([query_points, exp_times], -1)

            if wavelengths is None:
                fine_out = self._render(self.fine_model, query_points_time, rays_d, rays_o, z_vals_combined)
            else:
                fine_out = self._render(self.fine_model, query_points_time, rays_d, rays_o, z_vals_combined, wavelengths, instruments)

            # Store outputs.
            outputs['z_vals_hierarchical'] = z_hierarch
            outputs['fine_image'] = fine_out['image']
            image = fine_out['image']
            absorption = fine_out['regularizing_quantity']
            weights = fine_out['weights']

        else:
            outputs['z_vals_hierarchical'] = z_vals
            outputs['fine_image'] = coarse_out['image']
            image = coarse_out['image']
            absorption = coarse_out['regularizing_quantity']
            weights = coarse_out['weights']  

        # compute image of absorption
        absorption_map = (1 - absorption).sum(-1)
        # compute regularization of absorption
        distance = query_points.pow(2).sum(-1).pow(0.5)
        height_map = (weights * distance).sum(-1)
        # penalize absorption past 1.2 solar radii
        regularization = self.regularization(distance, absorption)

        # Store outputs.
        outputs['image'] = image
        outputs['height_map'] = height_map
        outputs['absorption_map'] = absorption_map
        outputs['regularization'] = regularization
        fine_out = None
        coarse_out = None
        image = None
        absorption = None
        weights = None
        distance = None
        height_map = None
        absorption_map = None
        regularization = None
        return outputs

    def forward_points(self, query_points):
        flat_points = query_points.view(-1, 4)
        raw_out = self.fine_model(flat_points)
        return raw_out

    def _render(self, model, query_points, rays_d, rays_o, z_vals):
        query_points_shape = query_points.shape[:-1]
        # Flattten query points for model input
        flat_query_points = query_points.view(-1, 4)
        # pass through the  model 
        raw = model(flat_query_points)
        # reshape 
        raw = raw.reshape(*query_points_shape, raw.shape[-1])
        # Perform differentiable volume rendering to re-synthesize the filtergrams.
        state = {'raw': raw, 'z_vals': z_vals, 'rays_d': rays_d, 'rays_o': rays_o, 'query_points': query_points}
        out = self.raw2outputs(**state)
        return out

    def raw2outputs(self, **kwargs):
        raise NotImplementedError("This method should be implemented in a subclass")


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod
