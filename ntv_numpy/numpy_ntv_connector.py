# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `numpy_ntv_connector` module is part of the `ntv-numpy.ntv_numpy` package
([specification document](
https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)).

A NtvConnector is defined by:
- clas_obj: str - define the class name of the object to convert
- clas_typ: str - define the NTVtype of the converted object
- to_obj_ntv: method - converter from JsonNTV to the object
- to_json_ntv: method - converter from the object to JsonNTV

It contains the child classes of `NTV.json_ntv.ntv.NtvConnector` abstract class:
    - `NarrayConnec`: 'narray' connector for np.ndarray data
    - `NdarrayConnec`: 'ndarray'   connector for Ndarray data
    - `XndarrayConnec`: 'xndarray' connector for Xndarray data
    - `XdatasetConnec`: 'xdataset' connector for Xdataset data
"""

from json_ntv import NtvConnector

from ntv_numpy.ndarray import Ndarray
from ntv_numpy.xndarray import Xndarray
from ntv_numpy.xdataset import Xdataset


class NarrayConnec(NtvConnector):
    """NTV connector for Numpy ndarray."""

    clas_obj = "ndarray"
    clas_typ = "narray"

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs):
        """convert json ntv_value into a np.ndarray.

        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with
        non Numpy ntv_type into data with python type"""
        return Ndarray.read_json(ntv_value, **kwargs).darray

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        """convert a np.ndarray (value, name, type) into NTV json (json-value, name, ntv_type).

        *Parameters*

        - **typ** : string (default None) - ntv_type of the np.ndarray object,
        - **name** : string (default None) - name of the ndarray object
        - **value** : np.ndarray value
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **format** : string (default 'full') - representation format of the ndarray,
        - **encoded** : Boolean (default False) - json-value if False else json-text
        - **header** : Boolean (default True) - including ndarray type
        """
        option = {
            "format": "full",
            "header": True,
            "encoded": False,
            "notype": False,
            "noshape": True,
            "novalue": False,
        } | kwargs
        if option["format"] not in ["full", "complete"]:
            option["noshape"] = False
        option["header"] = False
        return (Ndarray(value).to_json(**option), name, "narray")


class NdarrayConnec(NtvConnector):
    """NTV connector for Ndarray."""

    clas_obj = "Ndarray"
    clas_typ = "ndarray"

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs):
        """convert json ntv_value into a Ndarray.

        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with
        non-Numpy ntv_type into data with python type"""
        return Ndarray.read_json(ntv_value, **kwargs)

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        """convert a Ndarray (value, name, type) into NTV json (json-value, name, ntv_type).

        *Parameters*

        - **typ** : string (default None) - ntv_type of the ndarray object,
        - **name** : string (default None) - name of the ndarray object
        - **value** : Ndarray value (or np.ndarray value)
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **format** : string (default 'full') - representation format of the ndarray,
        - **encoded** : Boolean (default False) - json-value if False else json-text
        - **header** : Boolean (default True) - including ndarray type
        """
        option = {
            "format": "full",
            "header": True,
            "encoded": False,
            "notype": False,
            "noshape": True,
            "novalue": False,
        } | kwargs
        if option["format"] not in ["full", "complete"]:
            option["noshape"] = False
        return (Ndarray(value).to_json(**option), name, "ndarray")


class XndarrayConnec(NtvConnector):
    """NTV connector for Xndarray."""

    clas_obj = "Xndarray"
    clas_typ = "xndarray"

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs):
        """convert json ntv_value into a Xndarray.

        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with
        non-umpy ntv_type into Xndarray with python type
        """
        #print(ntv_value)
        return Xndarray.read_json(ntv_value, **kwargs)

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        """convert a Xndarray (value) into NTV json (json-value, name, ntv_type).

        *Parameters*

        - **typ** : string (default None) - not used,
        - **name** : string (default None) - not used
        - **value** : Xndarray values
        - **encoded** : Boolean (default False) - json-value if False else json-text
        - **header** : Boolean (default True) - including xndarray type
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **format** : string (default 'full') - representation format of the ndarray,
        - **extension** : string (default None) - type extension
        """
        option = {
            "notype": False,
            "extension": None,
            "format": "full",
            "noshape": True,
            "header": True,
            "encoded": False,
            "novalue": False,
            "noname": False,
        } | kwargs
        if option["format"] not in ["full", "complete"]:
            option["noshape"] = False
        option["header"] = False
        return (value.to_json(**option), name, "xndarray")


class XdatasetConnec(NtvConnector):
    """NTV connector for Xdataset."""

    clas_obj = "Xdataset"
    clas_typ = "xdataset"

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs):  # reindex=True, decode_str=False):
        """convert json ntv_value into a Xdataset.

        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with
        non-Numpy ntv_type into Xdataset with python type
        """
        return Xdataset.read_json(ntv_value, **kwargs)

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        """convert a Xdataset (value) into NTV json (json-value, name, ntv_type).

        *Parameters*

        - **typ** : string (default None) - not used,
        - **name** : string (default None) - not used
        - **value** : Xdataset entity
        - **encoded** : Boolean (default False) - json value if False else json text
        - **header** : Boolean (default True) - including 'xdataset' type
        - **notype** : list of Boolean (default list of None) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default False) - if True, without shape if dim < 1
        - **format** : list of string (default list of 'full') - representation format
        of the np.ndarray,
        """
        option = {
            "notype": False,
            "extension": None,
            "format": "full",
            "noshape": True,
            "header": False,
            "encoded": False,
            "novalue": False,
            "noname": True,
        } | kwargs
        if option["format"] not in ["full", "complete"]:
            option["noshape"] = False
        #option["noname"] = True
        return (value.to_json(**option), value.name, "xdataset")
