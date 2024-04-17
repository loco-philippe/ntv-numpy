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
    })
xdt = Xdataset.from_xarray(ds)
df = ds.to_dataframe().reset_index()

def to_tab_array(xdt, name, dims=None): 
    '''return a field np.array from a Xndarray defined by name
    
    parameters:
    
    - xdt: Xdataset to convert
    - name: string - name of the Xndarray to convert
    - dims: list of string (default None) - order of dimensions to apply
    '''
    dims = xdt.dimensions if not dims else dims
    n_shape = {nam: len(xdt[nam]) for nam in dims}
    dim_name = xdt.dims(name)
    if not set(dim_name) <= set(dims):
        return None
    add_name = [nam for nam in dims if not nam in dim_name]
    tab_name = add_name + dim_name
    
    til = 1 
    for nam in add_name:
        til *= n_shape[nam]
    shap = [n_shape[nam] for nam in tab_name]
    order = [dims.index(nam) for nam in tab_name]
    
    arr = xdt[name].darray
    
    old_order = list(range(len(order)))
    arr_tab = np.tile(arr, til).reshape(shap)
    return np.moveaxis(arr_tab, old_order, order).flatten()

#for name in xdt.dimensions[:]:
dimensions = ['x', 'y', 'z', 'year']
for name in xdt.names[:]:
    tab = to_tab_array(xdt, name, dimensions)
    if not tab is None: 
        print(np.all(np.array(df[name]) == tab), name)
        #print(tab)
    
"""x = np.array([10,20])
x_tab = np.tile(x, 3*2).reshape(3,2,2)
x_flat = np.moveaxis(x_tab,[0,1,2],[1,2,0]).flatten()

y = np.array(['a', 'b', 'c'])
y_tab = np.tile(y, 2*2).reshape(2,2,3)
y_flat = np.moveaxis(y_tab,[0,1,2],[0,2,1]).flatten()

year = np.array([2020, 2021])
year_tab = np.tile(year, 2*3).reshape(2,3,2)
year_flat = np.moveaxis(year_tab,[0,1,2],[0,1,2]).flatten()

foo2 = np.arange(12)
foo2_tab = np.tile(foo2, 1).reshape(3,2,2)
foo2_flat = np.moveaxis(foo2_tab,[0,1,2],[1,0,2]).flatten()

point = np.array(["pt1", "pt2", "pt3", "pt4", "pt5", "pt6"])
point_tab = np.tile(point, 2).reshape(2,2,3)
point_flat = np.moveaxis(point_tab,[0,1,2],[2,0,1]).flatten() """
