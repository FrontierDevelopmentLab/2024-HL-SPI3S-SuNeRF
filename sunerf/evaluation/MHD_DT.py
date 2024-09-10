import os
import shutil
from datetime import datetime

import matplotlib.axes as maxes
import numpy as np
import pandas as pd
from astropy import units as u 
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header
from tqdm import tqdm
from sunerf.model.mhd_model import MHDModel
import torch


from sunerf.rendering.density_temperature import DensityTemperatureRadiativeTransfer
from sunerf.evaluation.image_render import load_observer_meta
from sunerf.data.utils import sdo_cmaps, sdo_norms
from sunerf.evaluation.loader import ModelLoader
from sunerf.train.coordinate_transformation import spherical_to_cartesian
import matplotlib.pyplot as plt

from sunerf.data.mhd.psi_io import rdhdf_3d

# # TODO:  Avoid overwriting tNeRF rho, T data

data_path =  '/mnt/disks/data/MHD/rho/'
# data_path =  '/mnt/disks/data/MHD/t/'

#Open the H5 file in read mode
# with h5py.File(data_path+'rho001813.h5', 'r') as file:
#     print("Keys: %s" % file.keys())
#     a_group_key = list(file.keys())[0]
     
#     # Getting the data
#     data = list(file[a_group_key])
#     # print(data)
#     nrow = len(data)
#     ncol = len(data[0])
#     print(nrow, ncol)

r, th, phi, rho = rdhdf_3d(data_path+'rho001813.h5')
print(rho.shape)

    
#     # Iterate over the groups and datasets in the file
#     def print_structure(name, obj):
#         print(name)
#         if isinstance(obj, h5py.Group):
#             print("Group: ", name)
#         elif isinstance(obj, h5py.Dataset):
#             print("Dataset: ", name, " with shape ", obj.shape)
#         elif isinstance(obj, h5py.AttributeManager):
#             print("Attribute: ", name, " with value ", obj)
    
#     file.visititems(print_structure)
    
#         # Access the dataset (assuming the key is 'Data')
#     density_data = file['Data'][:]
#     print(density_data.shape)  # Print the shape of the density cube
   
#     # slice_data = data[0]  # This would access the first slice (1st of 630)
#     # print(slice_data.shape)  # This would print (292, 284)
   
    
# # import h5py

# # # Path to your HDF5 file (for example, for density data)
# # file_path = '/mnt/disks/data/MHD/rho/rho001813.h5'

# # # Open the H5 file in read mode
# # with h5py.File(file_path, 'r') as file:
# #     # List all keys in the HDF5 file
# #     keys = list(file.keys())
# #     print("Available keys in the file:", keys)
    
# #     # You can then access the data using one of the keys
# #     # For example, if 'Data' is the key
# #     if 'Data' in keys:
# #         data = file['Data'][:]
# #         print("Shape of the data:", data.shape)
# #     else:
# #         print("Key 'Data' not found in the file.")

    



