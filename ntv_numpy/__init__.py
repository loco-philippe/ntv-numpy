# -*- coding: utf-8 -*-
"""
***NTV-NumPy Package***

Created on March 2024

@author: philippe@loco-labs.io

This package contains the following classes and functions:

- `ntv-numpy.ntv_numpy.numpy_ntv_connector` :

    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.XDatasetConnec`
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.XndarrayConnec`
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.NdarrayConnec`
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.to_json`
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.read_json`

- `ntv-numpy.ntv_numpy.data_array` :
    
    - `ntv-numpy.ntv_numpy.data_array.Darray`
    - `ntv-numpy.ntv_numpy.data_array.Dfull`
    - `ntv-numpy.ntv_numpy.data_array.Dcomplete`
    
- `ntv-numpy.ntv_numpy.ndarray` :
    
    - `ntv-numpy.ntv_numpy.ndarray.Ndarray`
    - `ntv-numpy.ntv_numpy.ndarray.NpUtil`
    - `ntv-numpy.ntv_numpy.ndarray.NdarrayError`
    
- `ntv-numpy.ntv_numpy.xndarray` :
    
    - `ntv-numpy.ntv_numpy.xndarray.Xndarray`
    - `ntv-numpy.ntv_numpy.xndarray.NpUtil`
    - `ntv-numpy.ntv_numpy.xndarray.NdarrayError`
    
- `ntv-numpy.ntv_numpy.xdataset` :
    
    - `ntv-numpy.ntv_numpy.xdataset.Xdataset`
    
"""
#from pathlib import Path
from ntv_numpy.numpy_ntv_connector import XndarrayConnec, NdarrayConnec, read_json, to_json
from ntv_numpy.data_array import Dfull, Dcomplete, Darray
from ntv_numpy.ndarray import Ndarray, NpUtil
from ntv_numpy.xndarray import Xndarray
from ntv_numpy.xdataset import Xdataset
#import ntv_pandas.pandas_ntv_connector

#path = Path(ntv_numpy.numpy_ntv_connector.__file__).parent

#print('package :', __package__)