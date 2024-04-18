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
dfr = DataFrameConnec.xexport(xdt, json_name=False)

dfr = dfr.reset_index()
print(dfr.attrs['info']['structure'])
x = np.moveaxis(np.array(dfr['x']).reshape(2,3,2,3),
                [0,1,2,3], [0,1,3,2])[:, 0, 0, 0].flatten()
print(x)
y = np.moveaxis(np.array(dfr['y']).reshape(2,3,2,3),
                [0,1,2,3], [0,1,3,2])[0, :, 0, 0].flatten()
print(y)
z = np.moveaxis(np.array(dfr['z']).reshape(2,3,2,3),
                [0,1,2,3], [0,1,3,2])[0, 0, :, 0].flatten()
print(z)
point = np.moveaxis(np.array(dfr['point']).reshape(2,3,2,3),
                    [0,1,2,3], [0,1,3,2])[:, :, 0, 0].flatten()
print(point)
foo = np.moveaxis(np.array(dfr['foo']).reshape(2,3,2,3), 
                  [0,1,2,3], [0,1,3,2])[:, :, :, :].flatten()
print(foo)
print(xdt['foo'].darray)

