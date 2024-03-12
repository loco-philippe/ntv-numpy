# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:08:28 2024

@author: a lab in the Air
"""
from decimal import Decimal
import numpy as np
from datetime import date, time
import pandas as pd
from shapely.geometry import Point, LineString

import ntv_pandas as npd
from ntv_numpy import read_json, to_json
from ntv_numpy.numpy_ntv_connector import read_json_tab, to_json_tab
from ntv_numpy import NdarrayConnec, XndarrayConnec
from ntv_numpy import Darray, Dfull, Dcomplete, Ndarray, Xndarray, NpUtil, Xdataset

from json_ntv import NtvConnector, Ntv
SeriesConnec = NtvConnector.connector()['SeriesConnec']
DataFrameConnec = NtvConnector.connector()['DataFrameConnec']
nd_equals = Ndarray.equals

ex = {'var2': [['float[kg]', [2, 2], [10.1, 0.4, 3.4, 8.2]], ['x', 'y']]}
xn = Xndarray.read_json(ex)
print(Ntv.obj(xn))

example = [['int64[kg]', [[1, 2], [3,4]]],
           ['int', [[1, 2], [3,4]]],
           ['json', [1, 'two']],
           ['month', [1, 2]],
           ['base16', ['1F23', '236A5E']],
           ['duration', ['P3Y6M4DT12H30M5S', 'P3Y6M4DT12H30M']],
           ['uri', ['geo:13.4125,103.86673', 'geo:13.41,103.86']],
           ['email', ['John Doe <jdoe@mac.example>', 'Anna Doe <adoe@mac.example>']],
           ['ipv4', ['192.168.1.1', '192.168.2.5']]
           ]
for ex in example:
    arr = np.array(ex[1], dtype=NpUtil.dtype(ex[0]))
    for format in ['full', 'complete']:
        js = to_json(arr, ntv_typ=ex[0], format=format)
        #print(js)
        ex_rt = read_json(js, header=False)
                