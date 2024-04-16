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
'''ds = xr.Dataset(
    {"foo": (("x", "y", "year"), np.random.randn(2, 3, 2))},
    coords={
        "x": [10, 20],
        "y": ["a", "b", "c"],
        "year": [2020, 2021],
        "point": (("x", "y"), np.array(["pt1", "pt2", "pt3", "pt4", "pt5", "pt6"]).reshape(2,3)),
        "along_x": ("x", np.random.randn(2)),
        "scalar": 123,
    })'''
xdt = Xdataset.from_xarray(ds)
df = ds.to_dataframe().reset_index()

def tab_array(xdt, name, names=None): 
    '''return a field np.array from a dimension Xndarray defined by idx_name
    
    parameters:
    
    - xdt: Xdataset 
    '''
    #names = list(dts.dims)
    names = xdt.dimensions if not names else names
    #shape = xdt.shape
    #shape = [len(xdt[dim]) for dim in names]
    #n_shape = {key: val for key, val in zip(names, shape)}
    n_shape = {nam: len(xdt[nam]) for nam in names}
    #idx = names.index(idx_name)
    dim_name = xdt.dims(name)
    if not set(dim_name) <= set(names):
        return None
    add_name = [nam for nam in names if not nam in dim_name]
    tab_name = add_name + dim_name
    arr = np.array(xdt[name])
    til = 1 
    for nam in add_name:
        til *= n_shape[nam]
    shap = [n_shape[nam] for nam in tab_name]
    print(til, shap)
    arr_tab = np.tile(arr, til).reshape(shap)
    #order = [tab_name.index(nam) for nam in names]
    order = [names.index(nam) for nam in tab_name]
    print(order)
    return np.moveaxis(arr_tab, list(range(len(names))),order).flatten()

#for name in xdt.dimensions[:]:
names = ['x', 'y', 'z', 'year']
for name in xdt.names[:]:
    tab = tab_array(xdt, name, names)
    if not tab is None: 
        print(np.all(np.array(df[name]) == tab), name)
        print(tab)
    
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
