# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import pandas as pd
from ntv_numpy.xdataset import Xdataset
from ntv_numpy.xconnector import DataFrameConnec

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
    })
xdt = Xdataset.from_xarray(ds)
df = ds.to_dataframe().reset_index()
dfr = DataFrameConnec.xexport(xdt)
print(dfr.attrs['info']['structure'])
y = np.array(df['y']).reshape(2,3,2,3)[0, :, 0, 0].flatten()
print(y)
point = np.array(df['point']).reshape(2,3,2,3)[:, :, 0, 0].flatten()
print(point)
foo = np.moveaxis(np.array(df['foo']).reshape(2,3,2,3), 
                [0,1,2,3], [1,0,2,3]).flatten()
                #[0,1,2,3], [0,1,3,2]).flatten()

print(np.array(df['foo']).reshape(2,3,2,3).flatten())
print(df['foo'])

print(foo)
print(xdt['foo'].darray)