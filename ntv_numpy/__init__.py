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
    - `ntv-numpy.ntv_numpy.numpy_ntv_connector.NarrayConnec`

- `ntv-numpy.ntv_numpy.xconnector` :

    - `ntv-numpy.ntv_numpy.xconnector.PandasConnec`
    - `ntv-numpy.ntv_numpy.xconnector.XarrayConnec`
    - `ntv-numpy.ntv_numpy.xconnector.ScippConnec`
    - `ntv-numpy.ntv_numpy.xconnector.AstropyNDDataConnec`

- `ntv-numpy.ntv_numpy.data_array` :

    - `ntv-numpy.ntv_numpy.data_array.Darray`
    - `ntv-numpy.ntv_numpy.data_array.Dfull`
    - `ntv-numpy.ntv_numpy.data_array.Dcomplete`
    - `ntv-numpy.ntv_numpy.data_array.Dutil`

- `ntv-numpy.ntv_numpy.ndarray` :

    - `ntv-numpy.ntv_numpy.ndarray.Ndarray`
    - `ntv-numpy.ntv_numpy.ndarray.Nutil`
    - `ntv-numpy.ntv_numpy.ndarray.NdarrayError`

- `ntv-numpy.ntv_numpy.xndarray` :

    - `ntv-numpy.ntv_numpy.xndarray.Xndarray`

- `ntv-numpy.ntv_numpy.xdataset` :

    - `ntv-numpy.ntv_numpy.xdataset.Xdataset`
    - `ntv-numpy.ntv_numpy.xdataset.XdatasetInterface`
    - `ntv-numpy.ntv_numpy.xdataset.XdatasetCategory`

- `ntv-numpy.ntv_numpy.ndtype` :

    - `ntv-numpy.ntv_numpy.ndtype.Ndtype`


- `ntv-numpy.ntv_numpy.xarray_accessors` :

    - `ntv-numpy.ntv_numpy.xarray_accessors.NxrDatasetAccessor`
"""

from ntv_numpy.numpy_ntv_connector import XndarrayConnec, NdarrayConnec
from ntv_numpy.data_array import Dfull, Dcomplete, Dsparse, Drelative, Darray, Dutil
from ntv_numpy.ndarray import Ndarray, Nutil
from ntv_numpy.xndarray import Xndarray
from ntv_numpy.xdataset import Xdataset

import ntv_numpy.xarray_accessors as xarray_accessors

__all__ = [
    "XndarrayConnec",
    "NdarrayConnec",
    "Dfull",
    "Dcomplete",
    "Dsparse",
    "Drelative",
    "Darray",
    "Dutil",
    "Ndarray",
    "Nutil",
    "Xndarray",
    "Xdataset",
    "xarray_accessors",
]
