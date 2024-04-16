# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import pandas as pd
from ntv_numpy.xdataset import Xdataset

ds = xr.Dataset(
    {"foo": (("x", "y", "z", "year"), np.random.randn(2, 3, 3, 2))},
    coords={
        "x": [10, 20],
        "y": ["a", "b", "c"],
        "z": [1,2,3],
        "year": [2020, 2021],
        "point": (("x", "y"), np.array(["pt1", "pt2", "pt3", "pt4", "pt5", "pt6"]).reshape(2,3)),
        "along_x": ("x", np.random.randn(2)),
        "scalar": 123,
    },
)
xds = Xdataset.from_xarray(ds)
df = ds.to_dataframe().reset_index()

def tab_array(dts, idx_name): 
    '''return a field np.array from a dimension Xndarray
    
    parameters:
    
    - dts
    '''
    names = list(dts.dims)
    shape = list(dts.sizes.values())
    if isinstance(idx_name, int):
        idx = idx_name
        name = names[idx]
    else:
        idx = names.index(idx_name)
        name = idx_name
    arr = np.array(dts[name])
    return np.tile(np.repeat(arr, np.prod(shape[idx+1:])), np.prod(shape[:idx]))

for i in range(4):
    print(np.all(np.array(df[df.columns[i]]) == tab_array(ds, i)))
