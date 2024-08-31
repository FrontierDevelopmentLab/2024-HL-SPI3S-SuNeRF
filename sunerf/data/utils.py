from astropy import units as u
from astropy.visualization import ImageNormalize, LinearStretch
from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor
from sunpy.visualization.colormaps import cm
import numpy as np
from sunerf.baseline.reprojection import transform

sdo_img_norm = ImageNormalize(vmin=0, vmax=1, stretch=LinearStretch(), clip=True)

# !stretch is connected to NeRF!
sdo_norms = {171: ImageNormalize(vmin=0, vmax=8600, stretch=LinearStretch(), clip=False),
             193: ImageNormalize(vmin=0, vmax=9800, stretch=LinearStretch(), clip=False),
             195: ImageNormalize(vmin=0, vmax=9800, stretch=LinearStretch(), clip=False),
             211: ImageNormalize(vmin=0, vmax=5800, stretch=LinearStretch(), clip=False),
             284: ImageNormalize(vmin=0, vmax=5800, stretch=LinearStretch(), clip=False),
             304: ImageNormalize(vmin=0, vmax=8800, stretch=LinearStretch(), clip=False), }

psi_norms = {171: ImageNormalize(vmin=0, vmax=22348.267578125, stretch=LinearStretch(), clip=True),
             193: ImageNormalize(vmin=0, vmax=50000, stretch=LinearStretch(), clip=True),
             211: ImageNormalize(vmin=0, vmax=13503.1240234375, stretch=LinearStretch(), clip=True), }

so_norms = {304: ImageNormalize(vmin=0, vmax=300, stretch=LinearStretch(), clip=False),
            174: ImageNormalize(vmin=0, vmax=300, stretch=LinearStretch(), clip=False)}

sdo_cmaps = {171: cm.sdoaia171, 174: cm.sdoaia171, 193: cm.sdoaia193, 211: cm.sdoaia211, 304: cm.sdoaia304}


def loadAIAMap(file_path, resolution=1024, map_reproject=False, calibration:str='auto'):
    """Load and preprocess AIA file to make them compatible to ITI.


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels of 2.2 solar radii.
    map_reproject: apply preprocessing to remove off-limb (map to heliographic map and transform back to original view).

    Returns
    -------
    the preprocessed SunPy Map
    """
    s_map, _ = LoadMapEditor().call(file_path)
    if s_map.meta['QUALITY'] != 0:
        print(file_path)
    assert s_map.meta['QUALITY'] == 0, f'Invalid quality flag while loading AIA Map: {s_map.meta["QUALITY"]}'
    # s_map = NormalizeRadiusEditor(resolution).call(s_map)
    s_map = AIAPrepEditor(calibration=calibration).call(s_map)
    if map_reproject:
        s_map = transform(s_map, lat=s_map.heliographic_latitude,
                          lon=s_map.heliographic_longitude, distance=1 * u.AU)
    return s_map


def loadMap(file_path, resolution:int=None, map_reproject:bool=False, calibration:str='auto'):
    """Load and resample a FITS file (no pre-processing).


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels of 2.2 solar radii.

    Returns
    -------
    the preprocessed SunPy Map
    """
    s_map, _ = LoadMapEditor().call(file_path)
    if resolution:
        s_map = s_map.resample((resolution, resolution) * u.pix)
    s_map.meta['t_obs'] = s_map.meta['date-obs'] 

    s_map.meta['t_obs'] = s_map.meta['date-obs']

    return s_map


def loadMapStack(file_paths:list, resolution:int=1024, remove_nans:bool=True, map_reproject:bool=False, aia_preprocessing:bool=True,
                 calibration:str='auto', apply_norm:bool=True, percentile_clip:float=None)->np.array:
    """Load a stack of FITS files, resample ot specific resolution, and stack hem.

    Parameters
    ----------
    file_paths : list
        list of files to stack.
    resolution : int, optional
        target resolution, by default 1024
    remove_nans : bool, optional
        remove nans and infs, replace by 0, by default True
    map_reproject : bool, optional
        If to reproject the stack, by default False
    aia_preprocessing : bool, optional
        If to use aia preprocessing, by default True
    calibration : str, optional
        What type of AIA degradation fix to use, by default 'auto'
    apply_norm : bool, optional
        Whether to apply normalization after loading stack, by default True
    percentile_clip : float, optional
        If to apply a percentile clip. Number in percent: i.e. 0.25 means 0.25% NOT 25%, by default None

    Returns
    -------
    np.array
        returns AIA stack
    """    


    load_func = loadAIAMap if aia_preprocessing else loadMap
    s_maps = [load_func(file, resolution=resolution, map_reproject=map_reproject,
                        calibration=calibration) for file in file_paths]
    
    if apply_norm:
        stack = np.stack([sdo_norms[s_map.wavelength.value](s_map.data) for s_map in s_maps]).astype(np.float32)
    else:
        stack = np.stack([s_map.data for s_map in s_maps]).astype(np.float32)

    if remove_nans:
        stack[np.isnan(stack)] = 0
        stack[np.isinf(stack)] = 0

    if percentile_clip:
        for i in range(stack.shape[0]):
            percentiles = np.percentile(
                stack[i, :, :].reshape(-1), [100 - percentile_clip]
            )
            stack[i, :, :][stack[i, :, :] < 0] = 0
            stack[i, :, :][stack[i, :, :] > percentiles[0]] = percentiles[0]        

    return stack