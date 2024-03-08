# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:59:43 2024

@author: a lab in the Air
"""
import os
import datetime
import json
import configparser
from pathlib import Path
from decimal import Decimal

import pandas as pd
import numpy as np
from json_ntv import Ntv
#from abc import ABC, abstractmethod
from copy import copy
import json
from json_ntv import ShapelyConnec
from ntv_numpy.data_array import Dfull, Dcomplete, Darray

from json_ntv import NtvConnector
SeriesConnec = NtvConnector.connector().get('SeriesConnec')
DataFrameConnec = NtvConnector.connector().get('DataFrameConnec')


class Ndarray:

    @staticmethod
    def read_json(ntv_value, **kwargs):
        ''' convert json ntv_value into a ndarray.'''
        ntv_type = None
        shape = None
        match ntv_value[:-1]:
            case []: ...
            case [ntv_type, shape]: ...
            case [str(ntv_type)]: ...
            case [list(shape)]: ...
        darray = Darray.read_json(ntv_value[-1], dtype=NpUtil.dtype(ntv_type))
        darray.data = NpUtil.convert(ntv_type, darray.data, tojson=False)
        return darray.values.reshape(shape)

    @staticmethod
    def to_json(value, **kwargs):
        ''' convert a ndarray into json-value

        *Parameters*

        - **typ** : string (default None) - ntv_type of the ndarray object,
        - **value** : ndarray value
        - **noshape** : Boolean (default True) - if False, only shape if dim > 1
        - **notype** : Boolean (default False) - including data type if False
        - **format** : string (default 'full') - representation format of the ndarray,
        - **extension** : string (default None) - type extension
        '''
        option = {'typ': None, 'notype': False, 'extension': None, 'format': 'full',
                  'noshape': True} | kwargs
        if value is None:
            return None
        if len(value) == 0:
            return [[]]

        shape = list(value.shape)
        shape = None if len(shape) < 2 and option['noshape'] else shape
        val_flat = value.flatten() if shape else value

        ntv_type, ext = Ndarray.split_typ(option['typ'])
        ext = ext if ext else option['extension']
        dtype = val_flat.dtype.name
        dtype = val_flat[0].__class__.__name__ if dtype == 'object' else dtype
        ntv_type = NpUtil.ntv_type(dtype, ntv_type, ext)

        js_val = NpUtil.ntv_val(ntv_type, val_flat, option['format'])
        lis = [ntv_type if not option['notype'] else None, shape, js_val]
        return [val for val in lis if not val is None]

    @staticmethod
    def equals(npself, npother):
        '''return True if all elements are equals and dtype are equal'''
        if not (isinstance(npself, np.ndarray) and isinstance(npother, np.ndarray)):
            return False
        if npself.dtype != npother.dtype:
            return False
        if npself.shape != npother.shape:
            return False
        if len(npself.shape) == 0:
            return True
        if len(npself) != len(npother):
            return False
        if isinstance(npself[0], (np.ndarray, pd.Series, pd.DataFrame)):
            equal = {np.ndarray: Ndarray.equals,
                     pd.Series: SeriesConnec.equals,
                     pd.DataFrame: DataFrameConnec.equals}
            for a, b in zip(npself, npother):
                if not equal[type(npself[0])](a, b):
                    return False
            return True
        else:
            return np.array_equal(npself, npother)

    @staticmethod
    def add_ext(typ, ext):
        '''return extended typ'''
        ext = '[' + ext + ']' if ext else ''
        return '' if not typ else typ + ext

    @staticmethod
    def split_typ(typ):
        '''return a tuple with typ and extension'''
        if not isinstance(typ, str):
            return (None, None)
        spl = typ.split('[', maxsplit=1)
        return (spl[0], None) if len(spl) == 1 else (spl[0], spl[1][:-1])


class NpUtil:
    '''ntv-ndarray utilities.'''

    DATATION_DT = {'date': 'datetime64[D]', 'year': 'datetime64[Y]',
                   'yearmonth': 'datetime64[M]',
                   'datetime': 'datetime64[s]', 'datetime[ms]': 'datetime64[ms]',
                   'datetime[us]': 'datetime64[us]', 'datetime[ns]': 'datetime64[ns]',
                   'datetime[ps]': 'datetime64[ps]', 'datetime[fs]': 'datetime64[fs]',
                   'timedelta': 'timedelta64[s]', 'timedelta[ms]': 'timedelta64[ms]',
                   'timedelta[us]': 'timedelta64[us]', 'timedelta[ns]': 'timedelta64[ns]',
                   'timedelta[ps]': 'timedelta64[ps]', 'timedelta[fs]': 'timedelta64[fs]',
                   'timedelta[D]': 'timedelta64[D]', 'timedelta[Y]': 'timedelta64[Y]',
                   'timedelta[M]': 'timedelta64[M]'}
    DT_DATATION = {val: key for key, val in DATATION_DT.items()}

    CONNECTOR_DT = {'field': 'Series', 'tab': 'DataFrame'}
    DT_CONNECTOR = {val: key for key, val in CONNECTOR_DT.items()}

    PYTHON_DT = {'array': 'list', 'time': 'datetime.time',
                 'object': 'dict', 'null': 'NoneType', 'decimal64': 'Decimal',
                 'ndarray': 'ndarray'}
    DT_PYTHON = {val: key for key, val in PYTHON_DT.items()}

    OTHER_DT = {'boolean': 'bool', 'string': 'str'}
    DT_OTHER = {val: key for key, val in OTHER_DT.items()}

    LOCATION_DT = {'point': 'Point',
                   'line': 'LinearRing', 'polygon': 'Polygon'}
    DT_LOCATION = {val: key for key, val in LOCATION_DT.items()}

    NUMBER_DT = {'json': 'object', 'number': None, 'month': 'int', 'day': 'int',
                 'wday': 'int', 'yday': 'int', 'week': 'hour', 'minute': 'int',
                 'second': 'int'}
    STRING_DT = {'base16': 'str', 'base32': 'str', 'base64': 'str',
                 'period': 'str', 'duration': 'str', 'jpointer': 'str',
                 'uri': 'str', 'uriref': 'str', 'iri': 'str', 'iriref': 'str',
                 'email': 'str', 'regex': 'str', 'hostname': 'str', 'ipv4': 'str',
                 'ipv6': 'str', 'file': 'str', 'geojson': 'str', }
    FORMAT_CLS = {'full': Dfull, 'complete': Dcomplete}

    @staticmethod
    def convert(ntv_type, nda, tojson=True):
        ''' convert ndarray with external NTVtype.

        *Parameters*

        - **ntv_type** : string - NTVtype deduced from the ndarray name_type and dtype,
        - **nda** : ndarray to be converted.
        - **tojson** : boolean (default True) - apply to json function
        '''
        if tojson:
            match ntv_type:
                case dat if dat in NpUtil.DATATION_DT:
                    return nda.astype(NpUtil.DATATION_DT[dat]).astype(str)
                case 'bytes':
                    return nda.astype('bytes').astype(str)
                case 'time':
                    return nda.astype(str)
                case 'decimal64':
                    return nda.astype(float)
                case 'geojson':
                    return np.frompyfunc(ShapelyConnec.to_geojson, 1, 1)(nda)
                case _:
                    return nda
        else:
            match ntv_type:
                case None:
                    return nda
                case dat if dat in NpUtil.DATATION_DT:
                    return nda.astype(NpUtil.DATATION_DT[dat])
                case std if std in NpUtil.OTHER_DT:
                    return nda.astype(NpUtil.OTHER_DT[std])
                case 'time':
                    return np.frompyfunc(datetime.time.fromisoformat, 1, 1)(nda)
                case 'decimal64':
                    return np.frompyfunc(Decimal, 1, 1)(nda)
                case 'ndarray':
                    return np.frompyfunc(Ndarray.read_json, 1, 1)(nda)
                case python if python in NpUtil.PYTHON_DT:
                    return nda.astype('object')
                case connec if connec in NpUtil.CONNECTOR_DT:
                    return np.fromiter([NtvConnector.uncast(nd, None, connec)[0]
                                        for nd in nda], dtype='object')
                case 'point' | 'line' | 'polygon' | 'geometry':
                    return np.frompyfunc(ShapelyConnec.to_geometry, 1, 1)(nda)
                case _:
                    return nda.astype(NpUtil.dtype(ntv_type))

    @staticmethod
    def ntv_val(ntv_type, nda, form):
        ''' convert a simple ndarray into NTV json-value.

        *Parameters*

        - **ntv_type** : string - NTVtype deduced from the ndarray, name_type and dtype,
        - **nda** : ndarray to be converted.
        - **form** : format of data ('full', 'complete', 'sparse', 'primary').
        '''
        if form == 'complete' and len(nda) < 2:
            raise NdarrayError(
                "complete format is not available with ndarray length < 2")
        Format = NpUtil.FORMAT_CLS[form]
        darray = Format(nda)
        ref = darray.ref
        coding = darray.coding
        match ntv_type:
            case 'ndarray':
                data = [Ndarray.to_json(nd) for nd in darray.data]
            case connec if connec in NpUtil.CONNECTOR_DT:
                data = [NtvConnector.cast(nd, None, connec)[0]
                        for nd in darray.data]
            case 'point' | 'line' | 'polygon' | 'geometry':
                data = np.frompyfunc(ShapelyConnec.to_coord, 1, 1)(darray.data)
            case _:
                data = NpUtil.convert(ntv_type, darray.data)
        return Format(data, ref=ref, coding=coding).to_json()

    @staticmethod
    def ntv_type(dtype, ntv_type, ext):
        ''' return NTVtype from dtype, additional type and extension.

        *Parameters*

        - **dtype** : string - dtype of the ndarray
        - **ntv_type** : string - additional type
        - **ext** : string - type extension
        '''
        DT_NTVTYPE = (NpUtil.DT_DATATION | NpUtil.DT_LOCATION |
                      NpUtil.DT_OTHER | NpUtil.DT_CONNECTOR | NpUtil.DT_PYTHON)
        if ntv_type:
            return Ndarray.add_ext(ntv_type, ext)
        match dtype:
            case dat if dat in DT_NTVTYPE:
                return Ndarray.add_ext(DT_NTVTYPE[dat], ext)
            case string if string[:3] == 'str':
                return Ndarray.add_ext('string', ext)
            case byte if byte[:5] == 'bytes':
                return Ndarray.add_ext('bytes', ext)
            case _:
                return Ndarray.add_ext(dtype, ext)

    @staticmethod
    def dtype(ntv_type):
        ''' return dtype from ntv_type'''
        DTYPE = (NpUtil.DATATION_DT | NpUtil.NUMBER_DT | NpUtil.OTHER_DT |
                 NpUtil.STRING_DT)
        OBJECT = NpUtil.LOCATION_DT | NpUtil.CONNECTOR_DT | NpUtil.PYTHON_DT
        type_base = Ndarray.split_typ(ntv_type)[0]
        if type_base in OBJECT:
            return 'object'
        return DTYPE.get(ntv_type, DTYPE.get(type_base, type_base))
