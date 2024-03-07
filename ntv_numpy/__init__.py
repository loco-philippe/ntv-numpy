# -*- coding: utf-8 -*-
"""
***NTV-NumPy Package***

Created on March 2024

@author: philippe@loco-labs.io

This package contains the following classes and functions:

- `ntv-numpy.ntv_numpy.numpy_ntv_connector` :

    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.XndarrayConnec`
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.NdarrayConnec`
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.NpUtil`
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.to_json`
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.read_json`

- `ntv-numpy.ntv_numpy.data_array` :
    
    - `ntv-numpy.ntv_numpy.data_array.Darray`
    - `ntv-numpy.ntv_numpy.data_array.Dfull`
    - `ntv-numpy.ntv_numpy.data_array.Dcomplete`
    
    
"""
#from pathlib import Path
from ntv_numpy.numpy_ntv_connector import XndarrayConnec, NdarrayConnec, read_json
from ntv_numpy.numpy_ntv_connector import to_json
from ntv_numpy.data_array import Dfull, Dcomplete, Darray
from ntv_numpy.ndarray import Ndarray
from ntv_numpy.xndarray import Xndarray
#import ntv_pandas.pandas_ntv_connector

#path = Path(ntv_numpy.numpy_ntv_connector.__file__).parent

#print('package :', __package__)