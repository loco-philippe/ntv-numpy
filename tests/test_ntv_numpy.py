# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `test_ntv_numpy` module contains the unit tests (class unittest) for the
`Darray`, `Ndarray` and `Xndarray` classes.
"""
import unittest

from decimal import Decimal
import numpy as np
from datetime import datetime, date, time
from pprint import pprint
from json_ntv import Ntv
import pandas as pd
from shapely.geometry import Point, LinearRing
import ntv_pandas as npd
from ntv_numpy import read_json, to_json
from ntv_numpy.numpy_ntv_connector import read_json_tab, to_json_tab
from ntv_numpy import NdarrayConnec, XndarrayConnec, Darray, Dfull, Dcomplete
from ntv_pandas import SeriesConnec, DataFrameConnec

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
            da = Darray.read_list(ex[0])
            self.assertEqual(da.__class__.__name__, ex[1])            
            self.assertEqual(len(da), len(ex[0]))            
            match ex[1]:
                case 'Dfull':
                    self.assertIsNone(da.ref)
                    self.assertTrue(np.array_equal(np.array(None), da.coding))
                    self.assertTrue(np.array_equal(da.data, da.values))           
                case 'Dcomplete':
                    da_full = Darray.read_list(example[index-1][0])
                    self.assertIsNone(da.ref)
                    self.assertFalse(np.array_equal(np.array(None), da.coding))
                    self.assertTrue(np.array_equal(da_full.values, da.values))   


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
            self.assertTrue(np.array_equal(np.array(None), da.coding))
            self.assertTrue(np.array_equal(da.data, da.values))           

        """for ex in example:
            da = Dfull(ex)
            print(type(da), len(da))
            print(da.data, da.ref, da.coding)
            print(da.values)"""
    
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
                  #[[LinearRing([[0, 0], [0, 1], [1, 1]]), LinearRing([[0, 0], [0, 10], [10, 10]])], 'object'],
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
                    self.assertTrue(np.array_equal(ex_rt, arr))            
                    self.assertEqual(ex_rt.dtype.name, arr.dtype.name)            
                    #print(np.array_equal(ex_rt, arr),  ex_rt, ex_rt.dtype)
        
    def test_ndarray_nested(self):    

        example =[[[[1,2], [3,4]], 'object'],
                  [[np.array([1, 2], dtype='int64'), np.array(['test1', 'test2'], dtype='str_')], 'object'],
                  [[pd.Series([1,2,3]), pd.Series([4,5,6])], 'object'],
                  [[pd.DataFrame({'::date': ['1964-01-01', '1985-02-05'], 
                                   'names::string': ['john', 'eric']}),
                             pd.DataFrame({'::date': ['1984-01-01', '1995-02-05'], 
                                              'names::string': ['anna', 'erich']})], 'object' ]
                  ]
        print()
        for ex in example:
            arr = np.fromiter(ex[0], dtype=ex[1])
            for format in ['full', 'complete']:
                js = to_json(arr, format=format)
                print(js)
                ex_rt = read_json(js, header=False)
                print(np.array_equal(ex_rt, arr),  ex_rt, ex_rt.dtype)
        
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
        print()
        for ex in example:
            arr = np.array(ex[1])
            for format in ['full', 'complete']:
                js = to_json(arr, typ=ex[0], format=format)
                print(js)
                ex_rt = read_json(js, header=False)
                ex_rt_head = read_json(js)
                print(np.array_equal(ex_rt, arr),  ex_rt_head)

if __name__ == '__main__':
    
    unittest.main(verbosity=2)
                                    