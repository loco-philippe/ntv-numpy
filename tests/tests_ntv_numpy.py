# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `test_ntv_numpy` module contains the unit tests (class unittest) for the
`Darray`, `Ndarray` and `Xndarray` classes.
"""

import unittest
from decimal import Decimal
from datetime import date, time

import requests
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import xarray as xr
import ntv_pandas

from ntv_numpy import Darray, Dfull, Dcomplete, Dsparse, Drelative, Dimplicit, Ndarray, Xndarray, Xdataset, Dutil
from ntv_numpy.xconnector import PandasConnec

ana = ntv_pandas.to_analysis  # uniquement pour le pre-commit
nd_equals = Dutil.equals
FILE = "https://raw.githubusercontent.com/loco-philippe/ntv-numpy/master/example/ex_ndarray.ntv"


class TestDarray(unittest.TestCase):
    """test Darray class"""

    def test_decode_json(self):
        """test decode_json"""
        parametres = ['uri', 'data', 'keys',
                      'leng', 'coef', 'sp_idx', 'custom']
        examples = [
            ('ert', ['uri']),
            ([1, 2, 3], ['data']),
            ([["orange", "pepper", "apple"], [
             2, 2, 0, 2, 2, 1, 0, 2]], ['data', 'keys']),
            ([["orange", "pepper", "orange", "apple"], [8, [2, 5, 6, -1]]],
             ['data', 'leng', 'sp_idx']),
            ([["orange", "pepper", "apple"], [8, [[0, 1, 0, 2], [2, 5, 6, -1]]]],
             ['data', 'leng', 'keys', 'sp_idx']),
            ([[10, 20, 30], [18, [2]]], ['data', 'leng', 'coef'])]
        for jsn, params in examples:
            result = Darray.decode_json(jsn)
            for param in parametres:
                if param in params:
                    self.assertTrue(result[param] is not None)
                else:
                    self.assertTrue(result[param] is None)

    def test_not_json(self):
        """test json conversion"""
        example = [
            [np.array([1, 2], dtype="int64"),
             np.array(["test1", "test2"], dtype="str_")]
            ]

        for ex in example:
            for forma in [Dfull, Dcomplete, Dsparse]:
                # print(ex, forma)
                dar = forma(ex)
                self.assertEqual(dar.coding, Darray.read_json(dar.to_json()).coding)
                self.assertTrue(dar.to_json(), Darray.read_json(dar.to_json()).to_json())

    def test_json(self):
        """test json conversion"""
        example = [
            [10, 20, 30, 30, 40, 30, 50, 50, 30, 10],
            [1, 2, 3],
            [[1, 2], [2, 3], 4, [2, 3]],
            [[1, 2], {'a': 2, 'b': 3}, 4, [2, 3]],
            [[1, 2], [2, 3, 4], [2, 3]]]

        for ex in example:
            for forma in [Dfull, Dcomplete, Dsparse]:
                # print(ex, forma)
                dar = forma(ex)
                self.assertEqual(dar.coding, Darray.read_json(dar.to_json()).coding)
                self.assertEqual(list(dar.values), list(Darray.read_json(dar.to_json()).values))
                self.assertEqual(list(dar.values), ex)

    def test_relative_implicit(self):
        """test relative format"""
        parent = Dcomplete([10, 20, 20, 30, 30, 40])
        example = [
            [100, 200, 200, 100, 100, 200]
        ]
        for ex in example:
            dr = Drelative(ex, ref=parent)
            self.assertEqual(Darray.read_json(dr.to_json(), ref=parent), dr)
            dr2 = Dimplicit(ex, ref=parent)
            self.assertEqual(Darray.read_json(dr2.to_json(), ref=parent), dr2)            
            dr3 = Dimplicit(dr2.codec, ref=parent)
            self.assertEqual(Darray.read_json(dr3.to_json(), ref=parent), dr3)
            self.assertTrue(list(dr3.values) == list(dr2.values) == list(dr.values) == ex)

    def test_cast(self):
        parent = Dcomplete([10, 20, 20, 30, 30, 40], dtype='int32')
        example = [
            np.array([100, 200, 200, 100, 100, 200])
        ]    
        for ex in example:
            dr = Drelative(ex, ref=parent)
            di = Dimplicit(ex, ref=parent)
            dc = Dcomplete(ex)
            self.assertTrue(np.array_equal(dr.values, di.values, equal_nan=False) and
                            np.array_equal(dc.values, di.values, equal_nan=False))
            self.assertEqual(di, Dimplicit(Dcomplete(Drelative(dc, ref=parent)), ref=parent))
            self.assertEqual(dr, Drelative(Dcomplete(Dimplicit(dr, ref=parent)), ref=parent))
            
            
    def test_darray_simple(self):
        """test Darray"""
        example = [
            ([1, 2], "Dfull"),
            ([[1, 2], [0, 1]], "Dcomplete"),
            ([[10, 20], [1, 2]], "Dfull"),
            ([[[10, 20], [1, 2]], [0, 1]], "Dcomplete"),
        ]

        for index, ex in enumerate(example):
            da = Darray.read_json(ex[0])
            self.assertEqual(da.__class__.__name__, ex[1])
            self.assertEqual(len(da), len(ex[0]))
            match ex[1]:
                case "Dfull":
                    self.assertIsNone(da.ref)
                    self.assertTrue(da.coding is None)
                    self.assertTrue(nd_equals(da.data, da.values))
                case "Dcomplete":
                    da_full = Darray.read_json(example[index - 1][0])
                    self.assertIsNone(da.ref)
                    self.assertTrue(da.coding is not None)
                    self.assertTrue(nd_equals(da_full.values, da.values))

    def test_darray_dtype(self):
        """test Darray"""
        self.assertEqual(
            Darray.read_json([1, "two"], dtype="object").to_json(), [1, "two"]
        )

    def test_darray_nested(self):
        """test Darray"""
        example = [
            np.array(
                [
                    np.array([1, 2], dtype="int64"),
                    np.array(["test1", "test2"], dtype="str_"),
                ],
                dtype="object",
            )
        ]

        for ex in example:
            da = Dfull(ex)
            # print(da)
            self.assertEqual(len(da), len(ex))
            self.assertIsNone(da.ref)
            self.assertTrue(da.coding is None)
            self.assertTrue(nd_equals(da.data, da.values))


class TestNdarray(unittest.TestCase):
    """test Ndarray class"""

    def test_update(self):
        """test Ndarray"""
        nda = Ndarray.read_json([[2, 2], [1, 2, 3, 4]])
        nda1 = Ndarray.read_json([[2, 2], [1, 2, 3, 4]])
        self.assertTrue(nda.update(Ndarray.read_json(["ex_nda2"])))
        self.assertTrue(nda.update(Ndarray.read_json([[2, 2], [1, 2, 3, 4]])))
        self.assertEqual(nda, nda1)

    def test_set_array_uri(self):
        """test Ndarray"""
        ndas = Ndarray.read_json([[2, 2], "uri"])
        self.assertFalse(ndas.set_array([1, 2, 3]))
        self.assertTrue(ndas.set_array([[1, 2], [4, 3]]))
        self.assertFalse(ndas.set_array([[1.1, 2], [4, 3]]))
        self.assertTrue(ndas.set_array([[10, 20], [40, 30]]))
        self.assertFalse(ndas.set_array([[10, 20], [40, 30], [40, 30]]))
        self.assertTrue(ndas.set_uri("uri", no_ntv_type=True))
        self.assertEqual(ndas, Ndarray.read_json([[2, 2], "uri"]))

    def test_ndarray_null(self):
        """test Ndarray"""
        example = [[[], None]]

        for ex in example:
            # print(ex[0], ex[1])
            arr = Ndarray(ex[0], ntv_type=ex[1])
            js = arr.to_json()
            # print(js)
            ex_rt = Ndarray.read_json(js)
            self.assertEqual(ex_rt, arr)
            self.assertEqual(js, ex_rt.to_json())
            ex_rt = Ndarray.read_json(js, convert=False)
            # print(js, ex_rt.to_json(format=format))
            self.assertEqual(js[":ndarray"][1], ex_rt.to_json()[":ndarray"][1])
            # print(np.array_equal(ex_rt, arr),  ex_rt, ex_rt.dtype)

    def test_ndarray_simple2(self):
        """test Ndarray"""
        example = [
            [[1, 2], "int64"],
            [[1, 2], None],
            [[True, False], "boolean"],
            # [['1+2j', 1], 'complex'],
            [["test1", "test2"], "string"],
            [["2022-01-01T10:05:21.0002", "2023-01-01T10:05:21.0002"], "datetime"],
            [["2022-01-01", "2023-01-01"], "date"],
            [["2022-01", "2023-01"], "yearmonth"],
            [["2022", "2023"], "year"],
            # [[1,2], 'timedelta[D]'],
            [[b"abc\x09", b"abc"], "base16"],
            [[time(10, 2, 3), time(20, 2, 3)], "time"],
            [[{"one": 1}, {"two": 2}], "object"],
            [[None, None], "null"],
            [[Decimal("10.5"), Decimal("20.5")], "decimal64"],
            [[Point([1, 2]), Point([3, 4])], "point"],
            # [[Ntv.obj({':point':[1,2]}), NtvSingle(12, 'noon', 'hour')], 'ntv'],
            [
                [
                    LineString([[0, 0], [0, 1], [1, 1], [0, 0]]),
                    LineString([[0, 0], [0, 10], [10, 10], [0, 0]]),
                ],
                "line",
            ],
        ]

        for ex in example:
            # print(ex[0], ex[1])
            arr = Ndarray(ex[0], ntv_type=ex[1])
            for forma in ["full", "complete", "sparse"]:
                if forma != "sparse" or (len(arr) > 1 and not np.all(arr.darray==np.full(len(arr), arr[0]))): 
                    js = arr.to_json(format=forma)
                    # print(js, forma)
                    ex_rt = Ndarray.read_json(js)
                    self.assertTrue(ex_rt.shape == arr.shape == [2])
                    self.assertEqual(ex_rt, arr)
                    self.assertEqual(js, ex_rt.to_json(format=forma))
                    ex_rt = Ndarray.read_json(js, convert=False)
                    # print(js, ex_rt.to_json(format=format))
                    self.assertEqual(
                        js[":ndarray"][1], ex_rt.to_json(format=forma)[
                            ":ndarray"][1]
                    )
                    # print(np.array_equal(ex_rt, arr),  ex_rt, ex_rt.dtype)
            if len(ex[0]) == 2:
                arr = Ndarray(ex[0], ntv_type=ex[1], shape=[2, 1])
                for forma in ["full", "complete", "sparse"]:
                    if forma != "sparse" or not np.all(arr.darray==np.full(len(arr), arr[0])): 
                        # print(ex, format)
                        js = arr.to_json(format=forma)
                        # print(js)
                        ex_rt = Ndarray.read_json(js)
                        self.assertEqual(ex_rt, arr)

    def test_ndarray_nested2(self):
        """test Ndarray"""
        example = [
            [[[1, 2], [3, 4, 5]], "array"],
            [
                [
                    np.array([1, 2], dtype="int64"),
                    np.array(["test1", "test2"], dtype="str_"),
                ],
                "narray",
            ],
            [[pd.Series([1, 2, 3]), pd.Series([4, 5, 6])], "field"],
            [
                [
                    pd.DataFrame(
                        {
                            "::date": pd.Series([date(1964, 1, 1), date(1985, 2, 5)]),
                            "names": ["john", "eric"],
                        }
                    ),
                    pd.DataFrame(
                        {
                            "::date": pd.Series([date(1984, 1, 1), date(1995, 2, 5)]),
                            "names": ["anna", "erich"],
                        }
                    ),
                ],
                "tab",
            ],
        ]
        for ex in example:
            arr = Ndarray(ex[0], shape=[2], ntv_type=ex[1])
            for forma in ["full", "complete", "sparse"]:
                js = arr.to_json(format=forma)
                # print(js, ex, forma)
                ex_rt = Ndarray.read_json(js)
                # print(js, ex, forma, ex_rt, arr) #!!!
                self.assertEqual(ex_rt, arr)
                # print(nd_equals(ex_rt, arr),  ex_rt, ex_rt.dtype)

    def test_ndarray_ntvtype2(self):
        """test Ndarray"""
        example = [
            ["int64[kg]", [[1, 2], [3, 4]]],
            ["int", [[1, 2], [3, 4]]],
            ["json", [1, "two"]],
            ["month", [1, 2]],
            ["base16", ["1F23", "236A5E"]],
            ["duration", ["P3Y6M4DT12H30M5S", "P3Y6M4DT12H30M"]],
            ["uri", ["geo:13.4125,103.86673", "geo:13.41,103.86"]],
            ["email", ["John Doe <jdoe@mac.example>", "Anna Doe <adoe@mac.example>"]],
            # ['$org.propertyID', ['NO2', 'NH3']]
            ["ipv4", ["192.168.1.1", "192.168.2.5"]],
            [None, ["a", "s"]],
            # [None, 'uri'],
            ["float", "uri"],
        ]
        for ex in example:
            arr = Ndarray(ex[1], ntv_type=ex[0])
            for forma in ["full", "complete", "sparse"]:
                js = arr.to_json(format=forma)
                # print(js)
                ex_rt = Ndarray.read_json(js)
                # print(ex_rt)
                self.assertEqual(ex_rt, arr)

    '''def test_ndarray_uri2(self):
        """test Ndarray"""
        jsn = requests.get(FILE, allow_redirects=True, timeout=30).content.decode()
        # print(type(jsn), jsn)
        nda = Ndarray.read_json(jsn)
        # print(nda)
        self.assertEqual(
            nda, Ndarray.read_json({":ndarray": ["int64[kg]", [2, 2], [1, 2, 3, 4]]})
        )
        example = [
            ["uri", "int32", None],
            ["uri", None, None],
            ["uri", "int32", [2, 2]],
            ["uri", None, [2, 2]],
        ]
        for ex in example:
            nda = Ndarray(ex[0], ex[1], ex[2])
            self.assertEqual(Ndarray.read_json(nda.to_json()), nda)'''

    def test_relative_implicit(self):
        """test relative format"""
        parent = Dcomplete([10, 20, 20, 30, 30, 40])
        example = [
            [100, 200, 200, 100, 100, 200]
        ]
        for ex in example:
            jsn_ful = Ndarray(ex).to_json()
            jsn_rel = Ndarray(ex).to_json(ref=parent, format='relative')
            self.assertEqual(jsn_rel, {':ndarray': ['int32', [[100, 200], [0, 1, 0, 1]]]})            
            self.assertEqual(Ndarray.read_json(jsn_rel, ref=parent).to_json(), jsn_ful)
            jsn_imp = Ndarray(ex).to_json(ref=parent, format='implicit')
            self.assertEqual(jsn_imp, {':ndarray': ['int32', [100, 200, 100, 200]]})
            self.assertEqual(Ndarray.read_json(jsn_imp, ref=parent).to_json(), jsn_ful)


class TestXndarray(unittest.TestCase):
    """test Xndarray class"""

    def test_new_xndarray_simple(self):
        """test Xndarray"""
        example = [
            {"y:string": [["y1", "y2"]]},
        ]
        for ex in example:
            self.assertEqual(ex, Xndarray.read_json(ex).to_json(header=False))

        example = [
            {":xndarray": {":int64[kg]": [[10, 20]]}},
            {":xndarray": {":month": [[1, 2]]}},
            {":xndarray": {":ipv4": [["192.168.1.1", "192.168.2.5"]]}},
            {":xndarray": {":json": [[1, "two", {"three": 3}]]}},
            {":xndarray": {":base16": [[b"1F23", b"236A5E"]]}},
            {":xndarray": {
                ":uri": [["geo:13.4125,103.86673", "geo:13.41,103.86"]]}},
            {":xndarray": {":object": [FILE]}},
        ]
        for ex in example:
            # print(ex)
            self.assertEqual(ex, Xndarray.read_json(ex).to_json())
            xnd = Xndarray.read_json(ex)
            self.assertEqual(xnd, Xndarray.read_json(xnd.to_json()))

    def test_new_xndarray_dataset(self):
        """test Xndarray"""
        example = [
            [{"var1:object": [["x", "y"], FILE]}, "relative", "variable"],
            [{"var1": [["x", "y"], FILE]}, "relative", "variable"],
            [
                {"var2:float[kg]": [["x", "y"], [
                    2, 2], [10.1, 0.4, 3.4, 8.2]]},
                "absolute",
                "variable",
            ],
            [{"ranking:int": [["var1"], [2, 2], [1, 2, 3, 4]]},
                "absolute", "variable"],
            [{"x:string": [{"test": 21}, ["x1", "x2"]]}, "absolute", "namedarray"],
            [{"y:string": [["y1", "y2"]]}, "absolute", "namedarray"],
            [{"z:string": [["x"], ["z1", "z2"]]}, "absolute", "variable"],
            [{"x.mask:boolean": [[True, False]]}, "absolute", "additional"],
            [{"x.variance:float": [[0.1, 0.2]]}, "absolute", "additional"],
            [{"z.variance:float": [[0.1, 0.2]]}, "absolute", "additional"],
            [{"unit": "kg"}, "undefined", "meta"],
            [{"info": {"example": "everything"}}, "undefined", "meta"],
        ]

        for ex, mode, xtype in example:
            # print(ex)
            self.assertEqual(ex, Xndarray.read_json(ex).to_json(header=False))
            self.assertEqual(mode, Xndarray.read_json(ex).mode)
            self.assertEqual(xtype, Xndarray.read_json(ex).xtype)
            xa = Xndarray.read_json(ex)
            for format in ["full", "complete"]:
                # print(xa.to_json(format=format))
                # print(Xndarray.read_json(xa.to_json(format=format)))
                self.assertEqual(xa, Xndarray.read_json(
                    xa.to_json(format=format)))

        example2 = [
            {"var1:object": [["x", "y"], FILE]},
            {"var1": [["x", "y"], FILE]},
            {"var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]]},
            {"ranking": [["var1"], [2, 2], [1, 2, 3, 4]]},
            {"x": [{"test": 21}, ["x1", "x2"]]},
            {"y": [["y1", "y2"]]},
            {"z": [["x"], ["z1", "z2"]]},
            {"x.mask": [[True, False]]},
            {"x.variance": [[0.1, 0.2]]},
            {"z.variance": [[0.1, 0.2]]},
        ]
        for (ex, mode, xtype), ex2 in zip(example, example2):
            # print(ex, ex2)
            self.assertEqual(Xndarray.read_json(ex2).to_json(header=False), ex)

    def test_relative_implicit(self):
        """test relative format"""
        parent = Dcomplete([10, 20, 30, 30, 40, 50])
        example = [
            {"ranking:int": [["var1"], [2, 3], [1, 2, 1, 1, 2, 6]]}
        ]
        for ex in example:
            jsn = Xndarray.read_json(ex).to_json(format='sparse')
            xnda = Xndarray.read_json(jsn)
            xndb = Xndarray.read_json(xnda.to_json(ref=parent, format='relative'), ref=parent)            
            self.assertEqual(xnda, xndb)      
            xndc = Xndarray.read_json(xnda.to_json(ref=parent, format='implicit'), ref=parent)            
            self.assertEqual(xnda, xndc)      

class TestXdataset(unittest.TestCase):
    """test Xdataset class"""

    def test_xdataset_full(self):
        """test Xdataset"""
        example = {
            "test": {
                "var1": [["x", "y"], FILE],
                "var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]],
                "ranking": [["var1"], [2, 2], [1, 2, 3, 4]],
                "x": [{"test": 21}, ["x1", "x2"]],
                "y": [["y1", "y2"]],
                "z": [["x"], ["z1", "z2"]],
                "x.mask1": [[True, False]],
                "x.variance": [[0.1, 0.2]],
                "z.variance": [[0.1, 0.2]],
                "unit": [["kg"]],
                "info": {"example": "everything"},
            }
        }
        #notype = [True, False, True, True, True,
        #          True, True, True, True, True, True]
        notype = ['var1', 'ranking', 'x', 'y', 'z', 'x.mask1', 'x.variance',
         'z.variance', 'unit', 'info']
        xds = Xdataset.read_json(example)
        self.assertEqual(
            xds.to_json(notype=notype, noshape=True, header=False), example
        )
        self.assertEqual(xds.dimensions, ("x", "y"))
        self.assertEqual(
            xds.partition,
            {
                "coordinates": ["ranking", "z"],
                "data_vars": ["var1", "var2"],
                "uniques": ["unit"],
                "metadata": ["info"],
                "additionals": ["x.mask1", "x.variance", "z.variance"],
                "dimensions": ["x", "y"],
            },
        )

        xdim = Xdataset(xds[xds.dimensions])
        self.assertEqual(
            xdim.to_json(novalue=True, noshape=True),
            {
                ":xdataset": {
                    "x:string": [{"test": 21}, ["-"]],
                    "y:string": [["-"]],
                }
            },
        )

    def test_xdataset_info(self):
        """test Xdataset"""
        xd = Xdataset([Xndarray("example", np.array(["x1", "x2"]))], "test")
        self.assertEqual(
            xd.info,
            {
                "structure": {
                    "name": "test",
                    "xtype": "group",
                    "validity": "valid",
                    "data_arrays": ["example"],
                    "width": 1,
                },
                "data": {
                    "example": {
                        "length": 2,
                        "mode": "absolute",
                        "ntvtype": "string",
                        "shape": [2],
                        "xtype": "namedarray",
                    }
                },
            },
        )
        example = {
            "test": {
                "var1": [["x", "y"], FILE],
                "var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]],
                "ranking": [["var2"], [2, 2], [1, 2, 3, 4]],
                "x": [{"test": 21}, ["x1", "x2"]],
                "y": [["y1", "y2"]],
                "z": [["x"], ["z1", "z2"]],
                "z_bis": [["z1_bis", "z2_bis"]],
                "x.mask1": [[True, False], ["x"]],
                "x.variance": [[0.1, 0.2], ["x"]],
                "z.variance": [[0.1, 0.2], ["x"]],
                "unit": [["kg"]],
                "info": {"example": "everything"},
            }
        }

        xd = Xdataset.read_json(example)
        self.assertEqual(
            xd.info["structure"],
            {
                "name": "test",
                "xtype": "group",
                "data_vars": ["var1", "var2"],
                "data_arrays": ["z_bis"],
                "dimensions": ["x", "y"],
                "coordinates": ["ranking", "z"],
                "additionals": ["x.mask1", "x.variance", "z.variance"],
                "metadata": ["info"],
                "uniques": ["unit"],
                "validity": "undefined",
                "width": 12,
            },
        )

        del xd[("var1", "z_bis")]
        self.assertEqual(
            xd.info["structure"],
            {
                "name": "test",
                "xtype": "mono",
                "data_vars": ["var2"],
                "dimensions": ["x", "y"],
                "length": 4,
                "coordinates": ["ranking", "z"],
                "additionals": ["x.mask1", "x.variance", "z.variance"],
                "metadata": ["info"],
                "uniques": ["unit"],
                "validity": "valid",
                "width": 10,
            },
        )

        example = {
            "test": {
                "var1:float[m3]": [["x", "y"], [2, 2], "path/var1.ntv"],
                "var2:float[kg]": [["x", "y"], [2, 2], "path/var2.ntv"],
                "ranking": [["var2"], [2, 2], "path/ranking.ntv"],
                "x": [{"test": 21}, "path/x.ntv"],
                "y": ["path/y.ntv"],
                "z": [["x"], "path/z.ntv"],
                "z_bis": ["path/z_bis.ntv"],
                "x.mask1": [["x"], "path/x.mask1.ntv"],
                "x.variance": [["x"], "path/x.variance.ntv"],
                "z.variance": [["x"], "path/z.variance.ntv"],
                "info": {
                    "path": "https://github.com/loco-philippe/ntv-numpy/tree/main/example/"
                },
            }
        }
        xd = Xdataset.read_json(example)
        self.assertEqual(
            xd.info["structure"],
            {
                "name": "test",
                "xtype": "group",
                "data_vars": ["var1", "var2"],
                "data_arrays": ["z_bis"],
                "dimensions": ["x", "y"],
                "coordinates": ["ranking", "z"],
                "additionals": ["x.mask1", "x.variance", "z.variance"],
                "metadata": ["info"],
                "validity": "undefined",
                "width": 11,
            },
        )

    def test_xdataset_json_relative(self):
        """test json format"""
        simple = {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "f": [10, 20, 30, 40, 40],
            "c": [1, 1, 3, 4, 4],
            "d": [1, 1, 1, 4, 4],
            "e": [1, 1, 1, 1, 1],
        }
        df1 = pd.DataFrame(simple)
        xds = Xdataset.from_dataframe(df1)
        forma= {name:'complete' for name in xds.names}
        jsn = xds.to_json(notype='all', format=forma, header=False)
        self.assertEqual(jsn,
                         {'a': [[[1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]],
                          'b': [['a'], [[10, 20, 30, 40, 50], [0, 1, 2, 3, 4]]],
                          'f': [['a'], [[10, 20, 30, 40], [0, 1, 2, 3, 3]]],
                          'c': [['f'], [[1, 3, 4], [0, 0, 1, 2]]],
                          'd': [['c'], [[1, 4], [0, 0, 1]]],
                          'e': [[1]]})
        df2 = df1.copy(deep=True)
        df1['g'] = 'paris'
        df2['g'] = 'london'
        df3 = pd.concat([df1, df2])
        xds3 = Xdataset.from_dataframe(df3)
        forma= {name:'complete' for name in xds3.names}
        jsn3 = xds3.to_json(notype='all', header=False, format=forma)
        self.assertEqual(jsn['d'], jsn3['d'])
        self.assertEqual(jsn3['g'], [[['london', 'paris'], [0, 1]]])
        
class TestXdatasetXarrayScipp(unittest.TestCase):
    """test Scipp interface"""

    def test_xdataset_new_xarray(self):
        """test Scipp"""
        examples = [
            {
                "test": {
                    "var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]],
                    "unit": "kg",
                    "info": {"example": "everything"},
                    "ranking": [["var2"], [2, 2], [1, 2, 3, 4]],  # !!!
                    "x": [{"test": 21}, ["x1", "x2"]],
                    "y:date": [["2021-01-01", "2022-02-02"]],
                    "z": [["x"], ["z1", "z2"]],
                    # 'z_bis': [[['z1_bis', 'z2_bis']]],
                    "x.mask1": [[True, False]],
                    "x.variance": [[0.1, 0.2]],
                    "z.variance": [[0.1, 0.2]],
                }
            },
            {
                "test": {
                    "var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]],
                    "var2.variance": [[2, 2], [0.1, 0.2, 0.3, 0.4]],
                    "var2.mask1": [["x"], [True, False]],
                    "var2.mask2": [[2, 2], [True, False, False, True]],
                    "ranking:month": [["var2"], [2, 2], [1, 2, 3, 4]],  # !!!
                    "x:string": [{"test": 21}, ["23F0AE", "578B98"]],
                    "x.mask1": [[True, False]],
                    "y:date": [["2021-01-01", "2022-02-02"]],
                    "z:float": [["x"], [10, 20]],
                    # 'z_bis': [['z1_bis', 'z2_bis']],
                    "z.variance": [[0.1, 0.2]],
                    "unit": "kg",
                    "info": {"example": "everything"},
                }
            },
        ]
        for example in examples:
            xd = Xdataset.read_json(example)
            xd2 = Xdataset.from_xarray(xd.to_xarray(dataset=False))
            self.assertEqual(xd, xd2)
            xd2 = Xdataset.from_xarray(xd.to_xarray())
            self.assertEqual(xd, xd2)

        examples = [
            xr.DataArray(np.array([1, 2, 3, 4]).reshape([2, 2])),
            xr.Dataset(
                {"var": (["date", "y"], np.array(
                    [1, 2, 3, 4]).reshape([2, 2]))},
                coords={
                    "date": np.array(
                        ["2021-02-04", "2022-02-04"], dtype="datetime64[ns]"
                    ),
                    "y": np.array([10, 20]),
                },
            ),
            xr.Dataset(
                {"var": (["date", "y"], np.array(
                    [1, 2, 3, 4]).reshape([2, 2]))},
                coords={
                    "date": (
                        ["date"],
                        np.array(["2021-02-04", "2022-02-04"],
                                 dtype="datetime64[ns]"),
                        {"ntv_type": "date"},
                    ),
                    "y": np.array([10, 20]),
                },
            ),
            xr.Dataset(
                {"var": (["date", "y"], np.array(
                    [1, 2, 3, 4]).reshape([2, 2]))},
                coords={
                    "date": (
                        ["date"],
                        np.array(["2021-02-04", "2022-02-04"],
                                 dtype="datetime64[ns]"),
                        {"ntv_type": "date"},
                    ),
                    "y": np.array([Point([1, 2]), Point([3, 4])]),
                },
            ),
        ]

        for xar in examples:
            xd = Xdataset.from_xarray(xar)
            # print(xd.to_json())
            for dts in [False, True]:
                xar2 = xd.to_xarray(dataset=dts)
                xd2 = Xdataset.from_xarray(xar2)
                # print(xd2.to_json())
                self.assertEqual(xd, xd2)
                self.assertEqual(xd.to_json(), xd2.to_json())

    def test_xdataset_new_scipp(self):
        """test Scipp"""
        examples = [
            {  # "test": {
                "var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]],
                "var2.variance": [[2, 2], [0.1, 0.2, 0.3, 0.4]],
                "var2.mask1": [["x"], [True, False]],
                "var2.mask2": [[2, 2], [True, False, False, True]],
                "ranking:month": [["var2"], [2, 2], [1, 2, 3, 4]],  # !!!
                "x:string": [["23F0AE", "578B98"]],
                # "x.mask1": [[True, False]],
                "y:date": [["2021-01-01", "2022-02-02"]],
                "z:float": [["x"], [10, 20]],
                # 'z_bis': [['z1_bis', 'z2_bis']],
                "z.variance:float": [[0.1, 0.2]],
                # "unit": "kg",
                # "info": {"example": "everything"},
            },
            {
                "x:int32": [[10, 20]],
                "y:string": [["a", "b", "c"]],
                "z:int32": [[1, 2, 3]],
                "year:int32": [[2020, 2021]],
                "point:string": [
                    ["y", "x"],
                    [3, 2],
                    ["pt1", "pt2", "pt3", "pt4", "pt5", "pt6"],
                ],
                "along_x:float64": [["x"], [-1.18, -0.74]],
                "foo:float64": [["x", "y", "z", "year"], [2, 3, 3, 2], list(range(36))],
            },
        ]
        for example in examples:
            xd = Xdataset.read_json(example)
            xd2 = Xdataset.from_scipp(xd.to_scipp(dataset=False, info=False))
            self.assertEqual(xd, xd2)
            xd2 = Xdataset.from_scipp(xd.to_scipp(dataset=False))
            self.assertEqual(xd, xd2)
            xd2 = Xdataset.from_scipp(xd.to_scipp())
            self.assertEqual(xd, xd2)
            xd2 = Xdataset.from_scipp(xd.to_scipp(info=False))
            self.assertEqual(xd, xd2)

    def test_xdataset_new_mixte(self):
        """test Scipp"""
        examples = [
            {
                "test:xdataset": {
                    "var1": [["x", "y"], FILE],
                    "var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]],
                    "var2.variance": [[2, 2], [0.1, 0.2, 0.3, 0.4]],
                    "var2.mask1": [["x"], [True, False]],
                    "var2.mask2": [[2, 2], [True, False, False, True]],
                    "ranking:month": [["var2"], [2, 2], [1, 2, 3, 4]],  # !!!
                    "x:string": [["23F0AE", "578B98"]],  # , {'test': 21}],
                    "x.mask1": [[True, False]],
                    "y:date": [["2021-01-01", "2022-02-02"]],
                    "z:float": [["x"], [10, 20]],
                    "z_bis": [["z1_bis", "z2_bis"]],
                    "z.uncertainty": [[0.1, 0.2]],
                    "z.variance:float": [[0.1, 0.2]],
                    "info": {"example": "everything"},
                    "location": [["paris"]],
                }
            }
        ]
        for example in examples:
            xd = Xdataset.read_json(example)
            xd_sc = Xdataset.from_scipp(xd.to_scipp(dataset=False))
            xd_xr = Xdataset.from_xarray(xd.to_xarray(dataset=False))
            self.assertTrue(xd == xd_sc == xd_xr)
            xd_sc = Xdataset.from_scipp(xd.to_scipp())
            xd_xr = Xdataset.from_xarray(xd.to_xarray())
            self.assertTrue(xd == xd_sc == xd_xr)


class TestXdatasetPandas(unittest.TestCase):
    """test pandas interface"""

    def test_xdataset_dataframe(self):
        """test pandas interface"""
        ds = xr.Dataset(
            {"foo": (("x", "y", "z", "year"), np.random.randn(2, 3, 3, 2))},
            coords={
                "x": [10, 20],
                "y": ["a", "b", "c"],
                "z": [1, 2, 3],
                "year": [2020, 2021],
                "point": (
                    ("x", "y"),
                    np.array(["pt1", "pt2", "pt3", "pt4",
                             "pt5", "pt6"]).reshape(2, 3),
                ),
                "along_x": ("x", np.random.randn(2)),
                "scalar": 123,
            },
        )
        xdt = Xdataset.from_xarray(ds)
        df = ds.to_dataframe().reset_index()
        dimensions = ["x", "y", "z", "year"]
        for name in xdt.names[:]:
            # tab = xdt.to_tab_array(name, dimensions)
            tab = PandasConnec._to_np_series(xdt, name, dimensions)
            if tab is not None:
                self.assertTrue(np.all(np.array(df[name]) == tab), name)

        dfr = xdt.to_dataframe(ntv_type=True)
        xds = Xdataset.from_dataframe(dfr)
        self.assertEqual(xds, xdt)

    def test_xdataset_new_multidim(self):
        """test pandas interface"""

        example = {
            "test:xdataset": {
                "var1": [["x", "y"], FILE],
                "var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]],
                "var2.variance": [[2, 2], [0.1, 0.2, 0.3, 0.4]],
                "var2.mask1": [["x"], [True, False]],
                "var2.mask2": [[2, 2], [True, False, False, True]],
                "ranking:month": [["var2"], [2, 2], [1, 2, 3, 4]],  # !!!
                "x:string": [["23F0AE", "578B98"]],  # , {'test': 21}],
                "x.mask1": [[True, False]],
                "y:date": [["2021-01-01", "2022-02-02"]],
                "z:float": [["x"], [10, 20]],
                "z_bis": [["z1_bis", "z2_bis"]],
                "z.uncertainty": [[0.1, 0.2]],
                "z.variance:float": [[0.1, 0.2]],
                "info": {"example": "everything"},
                "location": "paris",
            }
        }
        xd = Xdataset.read_json(example)
        df = xd.to_dataframe()
        xd2 = Xdataset.from_dataframe(df)
        self.assertEqual(xd, xd2)
        example = {
            ":xdataset": {
                "var2:float[kg]": [["x", "y"], [2, 2], [10.1, 0.4, 3.4, 8.2]],
                "var2.variance": [[2, 2], [0.1, 0.2, 0.3, 0.4]],
                "var2.mask1": [["x"], [True, False]],
                "var2.mask2": [[2, 2], [True, False, False, True]],
                "ranking:month": [["x", "y"], [2, 2], [1, 2, 3, 4]],  # !!!
                "x:string": [["23F0AE", "578B98"]],  # , {'test': 21}],
                "x.mask1": [[True, False]],
                "y:date": [["2021-01-01", "2022-02-02"]],
                "z:float": [["x"], [10, 20]],
                "z.uncertainty": [[0.1, 0.2]],
                "z.variance:float": [[0.1, 0.2]],
                "location:string": [["paris"]],
            }
        }
        xd = Xdataset.read_json(example)
        df = xd.to_dataframe()
        xd2 = Xdataset.from_dataframe(df)
        self.assertEqual(xd, xd2)
        df3 = xd.to_dataframe(info=False)
        xd3 = Xdataset.from_dataframe(df3)
        self.assertEqual(xd2, xd3)

    def test_xdataset_multipart(self):
        """test pandas interface"""
        fruits = {
            "plants": [
                "fruit",
                "fruit",
                "fruit",
                "fruit",
                "vegetable",
                "vegetable",
                "vegetable",
                "vegetable",
            ],
            "plts": ["fr", "fr", "fr", "fr", "ve", "ve", "ve", "ve"],
            "quantity": [
                "1 kg",
                "10 kg",
                "1 kg",
                "10 kg",
                "1 kg",
                "10 kg",
                "1 kg",
                "10 kg",
            ],
            "product": [
                "apple",
                "apple",
                "orange",
                "orange",
                "peppers",
                "peppers",
                "carrot",
                "carrot",
            ],
            "price": [1, 10, 2, 20, 1.5, 15, 1.5, 20],
            "price level": ["low", "low", "high", "high", "low", "low", "high", "high"],
            "group": [
                "fruit 1",
                "fruit 10",
                "fruit 1",
                "veget",
                "veget",
                "veget",
                "veget",
                "veget",
            ],
            "id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
            "supplier": [
                "sup1",
                "sup1",
                "sup1",
                "sup2",
                "sup2",
                "sup2",
                "sup2",
                "sup1",
            ],
            "location": ["fr", "gb", "es", "ch", "gb", "fr", "es", "ch"],
            "valid": ["ok", "ok", "ok", "ok", "ok", "ok", "ok", "ok"],
        }
        df1 = pd.DataFrame(fruits)
        a_df = df1.npd.analysis(distr=True)
        xdt = Xdataset.from_dataframe(df1)
        df3 = xdt.to_dataframe(ntv_type=False).reset_index()
        df2 = df1.sort_values(a_df.partitions(mode="id")
                              [0]).reset_index(drop=True)
        df4 = df3.sort_values(a_df.partitions(mode="id")[0]).reset_index(drop=True)[
            df2.columns
        ]
        self.assertTrue(df4.equals(df2))

    def test_xdataset_unidim(self):
        """test pandas interface"""
        simple = {
            "a": [1, 2, 3, 4, 4],
            "b": [10, 20, 30, 40, 40],
            # 'b2': [10,20,30,40,40],
            "c": [1, 1, 3, 4, 4],
            "d": [1, 1, 1, 4, 4],
            "e": [1, 1, 1, 1, 1],
        }
        df1 = pd.DataFrame(simple)
        df3 = Xdataset.from_dataframe(df1).to_dataframe(
            ntv_type=False)[df1.columns]
        self.assertTrue(df3.equals(df1))

        simple = {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "b2": [10, 20, 30, 40, 40],
            "c": [1, 1, 3, 4, 4],
            "d": [1, 1, 1, 4, 4],
            "e": [1, 1, 1, 1, 1],
        }
        df1 = pd.DataFrame(simple)
        df3 = (
            Xdataset.from_dataframe(df1)
            .to_dataframe(ntv_type=False)
            .reset_index()[df1.columns]
        )
        self.assertTrue(df3.equals(df1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
