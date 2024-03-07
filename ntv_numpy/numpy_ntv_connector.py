# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:49:49 2024

@author: Philippe@loco-labs.io

The `numpy_ntv_connector` module is part of the `ntv-numpy.ntv_numpy` package
([specification document](
https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)).

A NtvConnector is defined by:
- clas_obj: str - define the class name of the object to convert
- clas_typ: str - define the NTVtype of the converted object
- to_obj_ntv: method - converter from JsonNTV to the object
- to_json_ntv: method - converter from the object to JsonNTV

It contains :

- functions `read_json` and `to_json` to convert JSON data and numpy entities

- the child classes of `NTV.json_ntv.ntv.NtvConnector` abstract class:
    - `NdarrayConnec`: 'ndarray'   connector
    - `XndarrayConnec`: 'xndarray' connector

- an utility class with static methods : `NpUtil`
"""

import os
import datetime
import json
import configparser
from pathlib import Path
import pandas as pd
import numpy as np
from decimal import Decimal

from json_ntv.ntv import Ntv, NtvList, NtvSingle
from json_ntv.ntv_util import NtvUtil
from json_ntv.ntv_connector import ShapelyConnec
from data_array import Dfull, Dcomplete, Darray
from ndarray import Ndarray

from json_ntv import NtvConnector   
SeriesConnec = NtvConnector.connector().get('SeriesConnec')
DataFrameConnec = NtvConnector.connector().get('DataFrameConnec')

def read_json(jsn, **kwargs):
    ''' convert JSON text or JSON Value to Numpy ndarray.

    *parameters*

    - **noadd** : boolean (default False) - If True additional data is not include
    - **header** : boolean (default True) - If True NTV entity with NTVtype is included
    '''
    option = {'noadd': False, 'header': True} | kwargs
    jso = json.loads(jsn) if isinstance(jsn, str) else jsn
    if isinstance(jso, dict) and len(jso) == 1:
        if 'xndarray' in list(jso)[0]:
            arr = XndarrayConnec.to_obj_ntv(list(jso.values())[0], **option)
            #if option['header']:
            #    return {list(jso)[0]: arr}
            return arr
        else:
            arr = NdarrayConnec.to_obj_ntv(list(jso.values())[0], **option)
            #if option['header']:
            #    return {arr[0] + ':' + list(jso)[0]: arr[1]}
            return arr 
    if isinstance(jso, list):
        option = {'noadd': False, 'header': False} | kwargs
        arr =  NdarrayConnec.to_obj_ntv(jso, **option)
        #if option['header']:
        #    return {arr[0]: arr[1]}
        return arr
    return None


def to_json(ndarray, **kwargs):
    ''' convert Numpy ndarray to JSON text or JSON Value.

    *parameters*
    - **encoded** : Boolean (default False) - json value if False else json text
    - **header** : Boolean (default True) - including ndarray or xndarray type
    - **notype** : Boolean (default False) - including data type if True
    - **name** : string (default None) - name of the ndarray
    - **typ** : string (default None) - type of the NTV object,
    - **extension** : string (default None) - type extension
    - **add** : dict (default None) - additional data :
        - **attrs** : dict (default none) - metadata
        - **dims** : array (default none) - name of axis
        - **coords** : dict (default none) - dict of 'xndarray'
    '''
    option = {'encoded': False, 'format': 'full', 'header': True, 
              'name': None, 'typ': None, 'extension':None, 'notype': False, 
              'add': None} | kwargs
    if ndarray.__class__.__name__ == 'ndarray' and not kwargs.get('add'):
        jsn, nam, typ = NdarrayConnec.to_json_ntv(ndarray, **option)
    else:
        jsn, nam, typ = XndarrayConnec.to_json_ntv(ndarray, **option)        
    name = nam if nam else ''
    if option['header'] or name:
        typ = ':' + typ if option['header'] else '' 
        jsn = {name + typ : jsn}
    if option['encoded']:
        return json.dumps(jsn)
    return jsn

def to_json_tab(ndarray, add=None, header=True):
    period = ndarray.shape
    dim = ndarray.ndim
    coefi = ndarray.size
    coef = []
    for per in period:
        coefi = coefi // per
        coef.append(coefi)
    
    add = add if add else {}   
    axe_n = add['dims'] if 'dims' in add else ['dim_' + str(i) for i in range(dim)]
    axe_v = [add['coords'][axe] for axe in axe_n if axe in add['coords']] if 'coords' in add else []
    axe_v = [axe[-1] for axe in axe_v] if len(axe_v) == len(axe_n) else [
                      list(range(period[i])) for i in range(dim)]
    jsn = {nam: [var, [coe]] for nam, var, coe in zip(axe_n, axe_v, coef)} | {
           'data::' + ndarray.dtype.name: ndarray.flatten().tolist()}
    if header:
        return {':tab': jsn}
    return jsn

def read_json_tab(js):
    js = js[':tab'] if ':tab' in js else js
    shape = []
    axes_n = []
    axes_v = []
    coef = []
    nda = None
    for name, val in js.items():
        if len(val) == 2 and isinstance(val[1], list) and len(val[1]) == 1:
            shape.append(len(val[0]))
            coef.append(val[1])
            axes_v.append(val[0])
            axes_n.append(name)            
        else:
            spl = name.split('::')
            nda = np.array(val, dtype=spl[1]) if len(spl)==2 else np.array(val)
    coef, shape, axes_n, axes_v = list(zip(*sorted(zip(coef, shape, axes_n, 
                                                       axes_v), reverse=True)))
    return (nda.reshape(shape), {'dims': list(axes_n), 
            'coords': {axe_n: [axe_v] for axe_n, axe_v in zip(axes_n, axes_v)}})

class NdarrayConnec(NtvConnector):

    '''NTV connector for pandas DataFrame.

    One static methods is included:

    - to_listidx: convert a DataFrame in categorical data
    '''

    clas_obj = 'ndarray'
    clas_typ = 'ndarray'

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs):
        ''' convert json ntv_value into a ndarray.'''
        ntv_type = None
        shape = None
        match ntv_value[:-1]:
            case [ntv_type, shape]:...
            case [ntv_type] if isinstance(ntv_type, str):...
            case [shape] if isinstance(shape, list):...
        darray = Darray.read_list(ntv_value[-1], dtype=NpUtil.dtype(ntv_type))
        darray.data = NpUtil.convert(ntv_type, darray.data, tojson=False)
        return darray.values.reshape(shape)

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        ''' convert a ndarray (value, name, type) into NTV json (json-value, name, type).

        *Parameters*

        - **typ** : string (default None) - type of the NTV object,
        - **name** : string (default None) - name of the NTV object
        - **value** : ndarray values
        - **notype** : Boolean (default False) - including data type if False
        - **extension** : string (default None) - type extension
        '''    
        option = {'notype': False, 'extension': None, 'format': 'full'} | kwargs
        if len(value) == 0:
            return ([[]], name, 'ndarray')
        typ, ext = Ndarray.split_typ(typ)
        ext = ext if ext else option['extension']
        dtype = value.dtype.name
        dtype = value[0].__class__.__name__ if dtype == 'object' else dtype
        ntv_type = NpUtil.ntv_type(dtype, typ, ext)
        
        shape = list(value.shape)
        shape = shape if len(shape) > 1 else None 

        form = option['format']    
        if shape:
            js_val   = NpUtil.ntv_val(ntv_type, value.flatten(), form)
        else:
            js_val   = NpUtil.ntv_val(ntv_type, value, form)
            
        lis = [ntv_type if not option['notype'] else None, shape, js_val]
        return ([val for val in lis if not val is None], name, 'ndarray')

    @staticmethod
    def to_jsonv(value):
        ''' convert a ndarray into json-value.'''    
        if len(value) == 0:
            return [[]]
        dtype = value.dtype.name
        dtype = value[0].__class__.__name__ if dtype == 'object' else dtype
        ntv_type = NpUtil.ntv_type(dtype, None, None)

        shape = list(value.shape)
        shape = shape if len(shape) > 1 else None 
        
        if shape:
            js_val   = NpUtil.ntv_val(ntv_type, value.flatten(), 'full')
        else:
            js_val   = NpUtil.ntv_val(ntv_type, value, 'full')

        lis = [ntv_type, shape, js_val]        
        return [val for val in lis if not val is None]
    
class XndarrayConnec(NtvConnector):

    '''NTV connector for xndarray.

    One static methods is included:

    - to_listidx: convert a DataFrame in categorical data
    '''

    clas_obj = 'xndarray'
    clas_typ = 'xndarray'

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs):  # reindex=True, decode_str=False):
        ''' convert json ntv_value into a ndarray.

        *Parameters*

        - **index** : list (default None) - list of index values,
        - **alias** : boolean (default False) - if True, alias dtype else default dtype
        - **annotated** : boolean (default False) - if True, NTV names are not included.
        '''        
        np_data = NdarrayConnec.to_obj_ntv(ntv_value['data'])
        add = {key: val for key, val in ntv_value.items()
                if key in ('attrs', 'dims', 'coords')}
        return (np_data, add) if add and not kwargs['noadd'] else np_data

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        ''' convert a xndarray (value, name, type) into NTV json (json-value, name, type).

        *Parameters*

        - **typ** : string (default None) - type of the NTV object,
        - **name** : string (default None) - name of the NTV object
        - **value** : ndarray values
        - **notype** : Boolean (default False) - including data type if False
        - **name** : string (default None) - name of the ndarray
        - **extension** : string (default None) - type extension
        - **add** : dict (default None) - additional data :
            - **attrs** : dict (default none) - metadata
            - **dims** : array (default none) - name of axis
            - **coords** : dict (default none) - dict of 'xndarray'
        '''
        add = kwargs.get('add')
        dims = add.get('dims') if add else None
        attrs = add.get('attrs') if add else None
        coords = add.get('coords') if add else None
        dic = {'data': NdarrayConnec.to_json_ntv(value, kwargs=kwargs)[0], 
               'dims': dims, 'coords': coords, 'attrs': attrs}
        return ({key: val for key, val in dic.items() if not val is None},
                name, 'xndarray')

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
    DT_DATATION = {val:key for key, val in DATATION_DT.items()}
    
    CONNECTOR_DT = {'field': 'Series', 'tab': 'DataFrame'}    
    DT_CONNECTOR = {val:key for key, val in CONNECTOR_DT.items()}
    
    PYTHON_DT = {'array': 'list', 'time': 'datetime.time',
                 'object': 'dict', 'null': 'NoneType', 'decimal64': 'Decimal',
                 'ndarray': 'ndarray'}
    DT_PYTHON = {val:key for key, val in PYTHON_DT.items()}

    OTHER_DT = {'boolean': 'bool', 'string': 'str'}
    DT_OTHER = {val:key for key, val in OTHER_DT.items()}
    
    LOCATION_DT = {'point': 'Point', 'line': 'LinearRing', 'polygon': 'Polygon'}
    DT_LOCATION = {val:key for key, val in LOCATION_DT.items()}
    
    NUMBER_DT = {'json': 'object', 'number': None, 'month': 'int', 'day': 'int',
                 'wday': 'int', 'yday': 'int', 'week': 'hour', 'minute': 'int',
                 'second': 'int'}
    STRING_DT = {'base16': 'str', 'base32': 'str', 'base64': 'str', 
                 'period': 'str', 'duration': 'str', 'jpointer': 'str',
                 'uri': 'str', 'uriref': 'str', 'iri': 'str', 'iriref': 'str',
                 'email': 'str', 'regex': 'str', 'hostname': 'str', 'ipv4': 'str',
                 'ipv6': 'str', 'file': 'str', 'geojson': 'str',}
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
                case dat if dat in NpUtil.DATATION_DT:
                    return nda.astype(NpUtil.DATATION_DT[dat])
                case std if std in NpUtil.OTHER_DT:
                    return nda.astype(NpUtil.OTHER_DT[std])                    
                case 'time':
                    return np.frompyfunc(datetime.time.fromisoformat, 1, 1)(nda)                   
                case 'decimal64':
                    return np.frompyfunc(Decimal, 1, 1)(nda)   
                case 'ndarray':
                    return np.frompyfunc(NdarrayConnec.to_obj_ntv, 1, 1)(nda) 
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
            raise NdarrayError("complete format is not available with ndarray length < 2")
        Format = NpUtil.FORMAT_CLS[form]
        darray = Format(nda)
        ref = darray.ref
        coding = darray.coding
        match ntv_type:
            case 'ndarray':
                #data = np.frompyfunc(NdarrayConnec.to_jsonv, 1, 1)(darray.data) 
                data = [NdarrayConnec.to_jsonv(nd) for nd in darray.data] 
            case connec if connec in NpUtil.CONNECTOR_DT:
                data = [NtvConnector.cast(nd, None, connec)[0] for nd in darray.data] 
            case 'point' | 'line' | 'polygon' | 'geometry':
                data = np.frompyfunc(ShapelyConnec.to_coord, 1, 1)(darray.data)
            case _:
                data = NpUtil.convert(ntv_type, darray.data)
        return Format(data, ref=ref, coding=coding).to_list()
    
    @staticmethod
    def ntv_type(dtype, typ, ext):
        ''' return NTVtype from dtype, additional type and extension.

        *Parameters*

        - **dtype** : string - dtype of the ndarray
        - **typ** : string - additional type
        - **ext** : string - type extension
        '''        
        DT_NTVTYPE = (NpUtil.DT_DATATION | NpUtil.DT_LOCATION | 
                      NpUtil.DT_OTHER | NpUtil.DT_CONNECTOR | NpUtil.DT_PYTHON)
        if typ:
            return Ndarray.add_ext(typ, ext)
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
        DTYPE = (NpUtil.DATATION_DT |  NpUtil.NUMBER_DT | NpUtil.OTHER_DT | 
                 NpUtil.STRING_DT)        
        OBJECT = NpUtil.LOCATION_DT | NpUtil.CONNECTOR_DT | NpUtil.PYTHON_DT      
        type_base = Ndarray.split_typ(ntv_type)[0]
        if type_base in OBJECT:
            return 'object'
        return DTYPE.get(ntv_type, DTYPE.get(type_base, type_base))


class NdarrayError(Exception):
    '''Multidimensional exception'''