# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `test_ntv_numpy` module contains the unit tests (class unittest) for the
`Darray`, `Ndarray` and `Xndarray` classes.
"""
import unittest
import requests

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

from json_ntv import NtvConnector, Ntv, NtvSingle, NtvList
SeriesConnec = NtvConnector.connector()['SeriesConnec']
DataFrameConnec = NtvConnector.connector()['DataFrameConnec']
nd_equals = Ndarray.equals

#read_json({':ndarray': ['int64', [1, 2]]})

class Test_Darray(unittest.TestCase):
    
    def test_darray_simple(self):
        
        example =[
            ([1, 2], 'Dfull'),
            ([[1, 2], [0, 1]], 'Dcomplete'),
            ([[10, 20], [1, 2]], 'Dfull'),
            ([[[10, 20], [1, 2]], [0, 1]], 'Dcomplete')
        ]

        for index, ex in enumerate(example):
            da = Darray.read_json(ex[0])
            self.assertEqual(da.__class__.__name__, ex[1])            
            self.assertEqual(len(da), len(ex[0]))            
            match ex[1]:
                case 'Dfull':
                    self.assertIsNone(da.ref)
                    self.assertTrue(nd_equals(np.array(None), da.coding))
                    self.assertTrue(nd_equals(da.data, da.values))           
                case 'Dcomplete':
                    da_full = Darray.read_json(example[index-1][0])
                    self.assertIsNone(da.ref)
                    self.assertFalse(nd_equals(np.array(None), da.coding))
                    self.assertTrue(nd_equals(da_full.values, da.values))   
        

    def test_darray_dtype(self):
        
        self.assertEqual(Darray.read_json([1, 'two'], dtype='object').to_json(),
                         [1, 'two'])
        
    def test_darray_nested(self):

        example =[
            np.array([np.array([1, 2], dtype='int64'), 
                       np.array(['test1', 'test2'], dtype='str_')],
                      dtype='object')
        ]

        for ex in example:
            da = Dfull(ex)
            self.assertEqual(len(da), len(ex))            
            self.assertIsNone(da.ref)
            self.assertTrue(nd_equals(np.array(None), da.coding))
            self.assertTrue(nd_equals(da.data, da.values))           
    
class Test_Ndarray(unittest.TestCase):    

    def test_ndarray_simple(self):    
        
        example =[[[1,2], 'int64'],
                  [[[1,2], [3,4]], 'int64'],
                  [[True, False], 'bool'],
                  [['1+2j', 1], 'complex'],
                  [['test1', 'test2'], 'str_'], 
                  [['2022-01-01T10:05:21.0002', '2023-01-01T10:05:21.0002'], 'datetime64'],
                  [['2022-01-01', '2023-01-01'], 'datetime64[D]'],
                  [['2022-01', '2023-01'], 'datetime64[M]'],
                  [['2022', '2023'], 'datetime64[Y]'],
                  #[[1,2], 'timedelta64[D]'],
                  [[b'abc\x09', b'abc'], 'bytes'],
                  [[time(10, 2, 3), time(20, 2, 3)], 'object'],
                  [[{'one':1}, {'two':2}], 'object'],
                  [[None, None], 'object'],
                  [[Decimal('10.5'), Decimal('20.5')], 'object'],
                  [[Point([1,2]), Point([3,4])], 'object'],
                  [[Ntv.obj({':point':[1,2]}), NtvSingle(12, 'noon', 'hour')], 'object'],
                  [[LineString([[0, 0], [0, 1], [1, 1], [0, 0]]), 
                    LineString([[0, 0], [0, 10], [10, 10], [0, 0]])], 'object'],
                  []]       
        for ex in example:
            if len(ex) == 0:
                self.assertEqual(to_json(np.array([])), {':ndarray': [[]]})
            else:
                arr = np.array(ex[0], dtype=ex[1])
                for format in ['full', 'complete']:
                    js = to_json(arr, format=format)
                    #print(js)
                    ex_rt = read_json(js, header=False)
                    self.assertTrue(nd_equals(ex_rt, arr))            
                    #print(np.array_equal(ex_rt, arr),  ex_rt, ex_rt.dtype)

    def test_ndarray_reverse_json(self):    
        
        example =[[[1,2], 'int64'],
                  [[[1,2], [3,4]], 'int64'],
                  [[True, False], 'bool'],
                  [['1+2j', 1], 'complex'],
                  [['test1', 'test2'], 'str_'], 
                  [['2022-01-01T10:05:21.0002', '2023-01-01T10:05:21.0002'], 'datetime64'],
                  [['2022-01-01', '2023-01-01'], 'datetime64[D]'],
                  [['2022-01', '2023-01'], 'datetime64[M]'],
                  [['2022', '2023'], 'datetime64[Y]'],
                  #[[1,2], 'timedelta64[D]'],
                  [[b'abc\x09', b'abc'], 'bytes'],
                  [[time(10, 2, 3), time(20, 2, 3)], 'object'],
                  [[{'one':1}, {'two':2}], 'object'],
                  [[None, None], 'object'],
                  [[Decimal('10.5'), Decimal('20.5')], 'object'],
                  [[Point([1,2]), Point([3,4])], 'object'],
                  [[LineString([[0, 0], [0, 1], [1, 1], [0, 0]]), 
                    LineString([[0, 0], [0, 10], [10, 10], [0, 0]])], 'object']
                  ]       
        for ex in example:
            arr = np.array(ex[0], dtype=ex[1])
            for format in ['full', 'complete']:
                js = to_json(arr, format=format)
                #print(js)
                ex_rt = read_json(js, header=False)
                self.assertEqual(js, to_json(ex_rt, format=format))         
                #print(np.array_equal(ex_rt, arr),  ex_rt, ex_rt.dtype)

    def test_ndarray_not_convert(self):    
        
        example =[[[time(10, 2, 3), time(20, 2, 3)], 'object'],
                  [[{'one':1}, {'two':2}], 'object'],
                  [[None, None], 'object'],
                  [[Decimal('10.5'), Decimal('20.5')], 'object'],
                  [[Point([1,2]), Point([3,4])], 'object'],
                  [[LineString([[0, 0], [0, 1], [1, 1], [0, 0]]), 
                    LineString([[0, 0], [0, 10], [10, 10], [0, 0]])], 'object']
                  ]       
        for ex in example:
            arr = np.array(ex[0], dtype=ex[1])
            for format in ['full', 'complete']:
                js = to_json(arr, format=format)
                #print(js)
                ex_rt = read_json(js, header=False, convert=False)
                #print(js, to_json(ex_rt, format=format))
                self.assertEqual(js[':ndarray'][1], to_json(ex_rt, format=format)[':ndarray'][1])         
                #print(js, to_json(ex_rt, format=format))
                
    def test_ndarray_shape(self):    
        
        example =[[[1,2], 'int64'],
                  [[True, False], 'bool'],
                  [['1+2j', 1], 'complex'],
                  [['test1', 'test2'], 'str_'], 
                  [['2022-01-01T10:05:21.0002', '2023-01-01T10:05:21.0002'], 'datetime64'],
                  [['2022-01-01', '2023-01-01'], 'datetime64[D]'],
                  [['2022-01', '2023-01'], 'datetime64[M]'],
                  [['2022', '2023'], 'datetime64[Y]'],
                  #[[1,2], 'timedelta64[D]'],
                  [[b'abc\x09', b'abc'], 'bytes'],
                  [[time(10, 2, 3), time(20, 2, 3)], 'object'],
                  [[{'one':1}, {'two':2}], 'object'],
                  [[None, None], 'object'],
                  [[Decimal('10.5'), Decimal('20.5')], 'object'],
                  [[Point([1,2]), Point([3,4])], 'object'],
                  [[LineString([[0, 0], [0, 1], [1, 1]]), 
                    LineString([[0, 0], [0, 10], [10, 10]])], 'object']
                  ]        
        for ex in example:
            arr = np.array(ex[0], dtype=ex[1]).reshape([2,1])
            for format in ['full', 'complete']:
                #print(ex, format)
                js = to_json(arr, format=format)
                #print(js)
                ex_rt = read_json(js, header=False)
                self.assertTrue(nd_equals(ex_rt, arr))            
                
    def test_ndarray_nested(self):    

        example =[[[[1,2], [3,4]], 'object'],
                  [[np.array([1, 2], dtype='int64'), np.array(['test1', 'test2'], dtype='str_')], 'object'],
                  [[pd.Series([1,2,3]), pd.Series([4,5,6])], 'object'],
                  [[pd.DataFrame({'::date': pd.Series([date(1964,1,1), date(1985,2,5)]), 
                                  'names': ['john', 'eric']}),
                    pd.DataFrame({'::date': pd.Series([date(1984,1,1), date(1995,2,5)]), 
                                  'names': ['anna', 'erich']})], 'object' ]
                  ]
        for ex in example:
            arr = np.fromiter(ex[0], dtype=ex[1])
            for format in ['full', 'complete']:
                js = to_json(arr, format=format)
                #print(js)
                ex_rt = read_json(js, header=False)
                self.assertTrue(nd_equals(ex_rt, arr))            
                #print(nd_equals(ex_rt, arr),  ex_rt, ex_rt.dtype)
        
    def test_ndarray_ntvtype(self):    

        example = [['int64[kg]', [[1, 2], [3,4]]],
                   ['int', [[1, 2], [3,4]]],
                   ['json', [1, 'two']],
                   ['month', [1, 2]],
                   ['base16', ['1F23', '236A5E']],
                   ['duration', ['P3Y6M4DT12H30M5S', 'P3Y6M4DT12H30M']],
                   ['uri', ['geo:13.4125,103.86673', 'geo:13.41,103.86']],
                   ['email', ['John Doe <jdoe@mac.example>', 'Anna Doe <adoe@mac.example>']],
                   #['$org.propertyID', ['NO2', 'NH3']]
                   ['ipv4', ['192.168.1.1', '192.168.2.5']]
                   ]
        for ex in example:
            arr = np.array(ex[1], dtype=NpUtil.dtype(ex[0]))
            for format in ['full', 'complete']:
                js = to_json(arr, typ=ex[0], format=format)
                #print(js)
                ex_rt = read_json(js, header=False)
                #print(ex_rt)
                self.assertTrue(nd_equals(ex_rt, arr))    
        
    def test_ndarray_uri(self):    
        file = 'https://raw.githubusercontent.com/loco-philippe/ntv-numpy/master/example/ex_ndarray.ntv'
        jsn = requests.get(file, allow_redirects=True).content.decode()
        nda = read_json(jsn)
        self.assertEqual(to_json(nda), {':ndarray': ['int64', [2, 2], [1, 2, 3, 4]]})

class Test_Xndarray(unittest.TestCase):    

    def test_xndarray_simple(self):    
        
        example =[
                  {'y': [['string', ['y1', 'y2']]]},
                  
                ]
        for ex in example:
            self.assertEqual(ex, Xndarray.read_json(ex).to_json(header=False))         
            
        example =[
                  {':xndarray': [['int64[kg]', [10, 20]]]},
                  {':xndarray': [['month', [1, 2]]]},
                  {':xndarray': [['ipv4', ['192.168.1.1', '192.168.2.5']]]},
                  {':xndarray': [['json', [1, 'two', {'three': 3}]]]},
                  {':xndarray': [['base16', ['1F23', '236A5E']]]},
                  {':xndarray': [['uri', ['geo:13.4125,103.86673', 'geo:13.41,103.86']]]},
                  {':xndarray': ['https://github.com/loco-philippe/ntv-numpy/tree/main/example/ex_ndarray.ntv']}   
                ]
        for ex in example:
            self.assertEqual(ex, Xndarray.read_json(ex).to_json())    
            xnd = Xndarray.read_json(ex)
            self.assertEqual(xnd, Xndarray.read_json(xnd.to_json()))    
            
    def test_xndarray_dataset(self):    
        
        example =[[{'var1': ['https://github.com/loco-philippe/ntv-numpy/tree/main/example/ex_ndarray.ntv', 
                                    ['x', 'y']]}, 'relative', 'variable'],
                  [{'var2': [['float[kg]', [2, 2], [10.1, 0.4, 3.4, 8.2]], ['x', 'y']]}, 'absolute', 'variable'],

                  [{'ranking': [['int32', [2, 2], [1, 2, 3, 4]], ['var1']]}, 'absolute', 'variable'],
                  [{'x': [['string', ['x1', 'x2']], {'test': 21}]}, 'absolute', 'dimension'],
                  [{'y': [['string', ['y1', 'y2']]]}, 'absolute', 'dimension'],
                  [{'z': [['string', ['z1', 'z2']], ['x']]}, 'absolute', 'variable'],
                  [{'x.mask': [['boolean', [True, False]], ['x']]}, 'absolute', 'additional'],
                  [{'x.variance': [['float64', [0.1, 0.2]], ['x']]}, 'absolute', 'additional'],
                  [{'z.variance': [['float64', [0.1, 0.2]], ['x']]}, 'absolute', 'additional'],
                  [{'unit': 'kg'}, 'undefined', 'metadata'],
                  [{'info': {'example': 'everything'}}, 'undefined', 'metadata'],
                ]
        
        for ex, mode, xtype in example:
            self.assertEqual(ex, Xndarray.read_json(ex).to_json(header=False)) 
            self.assertEqual(mode, Xndarray.read_json(ex).mode) 
            self.assertEqual(xtype, Xndarray.read_json(ex).xtype) 
            xa = Xndarray.read_json(ex)
            for format in ['full', 'complete']:
                #print(xa.to_json(format=format))
                #print(Xndarray.read_json(xa.to_json(format=format)))
                self.assertEqual(xa, Xndarray.read_json(xa.to_json(format=format)))      

        example2 =[{'var1': ['https://github.com/loco-philippe/ntv-numpy/tree/main/example/ex_ndarray.ntv', 
                            ['x', 'y']]},
                   {'var2': [['float[kg]', [2, 2], [10.1, 0.4, 3.4, 8.2]], ['x', 'y']]},
                   {'ranking': [[[2, 2], [1, 2, 3, 4]], ['var1']]},
                   {'x': [[['x1', 'x2']], {'test': 21}]},
                   {'y': [[['y1', 'y2']]]},
                   {'z': [[['z1', 'z2']], ['x']]},
                   {'x.mask': [[[True, False]], ['x']]},
                   {'x.variance': [[[0.1, 0.2]], ['x']]},
                   {'z.variance': [[[0.1, 0.2]], ['x']]},
                  ]
        for (ex, mode, xtype), ex2 in zip(example, example2):
            #print(ex, ex2)
            self.assertEqual(Xndarray.read_json(ex2).to_json(header=False), ex)

class Test_Xdataset(unittest.TestCase):    

    def test_xdataset_full(self):    
        
        example = {'test': {
            'var1': ['https://github.com/loco-philippe/ntv-numpy/tree/main/example/ex_ndarray.ntv', 
                     ['x', 'y']],
            'var2': [['float[kg]', [2, 2], [10.1, 0.4, 3.4, 8.2]], ['x', 'y']],
            'ranking': [[[2, 2], [1, 2, 3, 4]], ['var1']],
            'x': [[['x1', 'x2']], {'test': 21}],
            'y': [[['y1', 'y2']]],
            'z': [[['z1', 'z2']], ['x']],
            'x.mask': [[[True, False]], ['x']],
            'x.variance': [[[0.1, 0.2]], ['x']],
            'z.variance': [[[0.1, 0.2]], ['x']],
            'unit': 'kg',
            'info': {'example': 'everything'}}}
        
        notype = [True, False, True, True, True, True, True, True, True, True, True]
        xds = Xdataset.read_json(example)        
        self.assertEqual(xds.to_json(notype=notype, noshape=True, header=False), example)                                          
        self.assertEqual(xds.dimensions, ('x', 'y'))
        self.assertEqual(xds.partition, {'coordinates': ['ranking', 'z'],
         'data_vars': ['var1', 'var2'], 'metadata': ['info', 'unit'],
         'dimensions': ['x', 'y']})
        
        xdim = Xdataset(xds[xds.dimensions])
        self.assertEqual(xdim.to_json(novalue=True, noshape=True), {':xdataset': {
                         'x': [['string', ['-']], {'test': 21}],
                         'y': [['string', ['-']]]}})        
                                  

if __name__ == '__main__':    
    unittest.main(verbosity=2)
                                    