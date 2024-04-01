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

It contains :

- functions `read_json` and `to_json` to convert JSON data and numpy entities

- the child classes of `NTV.json_ntv.ntv.NtvConnector` abstract class:
    - `NdarrayConnec`: 'ndarray'   connector
    - `XndarrayConnec`: 'xndarray' connector

- an utility class with static methods : `NpUtil`
"""

import json
import numpy as np
from ntv_numpy.ndarray import Ndarray, NpUtil
from ntv_numpy.xndarray import Xndarray
from ntv_numpy.xdataset import Xdataset

from json_ntv import NtvConnector
SeriesConnec = NtvConnector.connector().get('SeriesConnec')
DataFrameConnec = NtvConnector.connector().get('DataFrameConnec')

def read_json(jsn, **kwargs):
    ''' convert JSON text or JSON Value to Numpy ndarray.

    *parameters*

    - **noadd** : boolean (default False) - If True additional data is not include
    - **header** : boolean (default True) - If True NTV entity with NTVtype is included
    - **convert** : boolean (default True) - If True, convert json data with 
    non Numpy ntv_type into Xndarray with python type
    '''
    option = {'noadd': False, 'header': True, 'convert': True} | kwargs
    jso = json.loads(jsn) if isinstance(jsn, str) else jsn
    if isinstance(jso, dict) and len(jso) == 1:
        if 'xndarray' in list(jso)[0]:
            arr = XndarrayConnec.to_obj_ntv(list(jso.values())[0], **option)
            return arr
        else:
            arr = NdarrayConnec.to_obj_ntv(list(jso.values())[0], **option)
            return arr
    if isinstance(jso, list):
        option = {'noadd': False, 'header': False} | kwargs
        arr =  NdarrayConnec.to_obj_ntv(jso, **option)
        return arr
    return None


def to_json(ndarray, **kwargs):
    ''' convert Numpy ndarray to JSON text or JSON Value.

    *parameters*
    - **encoded** : Boolean (default False) - json value if False else json text
    - **header** : Boolean (default True) - including ndarray or xndarray type
    - **notype** : Boolean (default False) - including data type if True
    - **noshape** : Boolean (default True) - if True, without shape if dim < 1
    - **novalue** : Boolean (default False) - including value if False
    - **name** : string (default None) - name of the ndarray
    - **typ** : string (default None) - type of the NTV object,
    - **format** : string (default 'full') - representation format of the ndarray,
    - **extension** : string (default None) - type extension
    - **add** : dict (default None) - additional data :
        - **attrs** : dict (default none) - metadata
        - **dims** : array (default none) - name of axis
        - **coords** : dict (default none) - dict of 'xndarray'
    '''
    option = {'encoded': False, 'format': 'full', 'header': True,
              'name': None, 'typ': None, 'extension':None, 'notype': False,
              'noshape': True, 'novalue': False, 'add': None} | kwargs
    if ndarray.__class__.__name__ == 'ndarray' and not kwargs.get('add'):
        jsn, nam, typ = NdarrayConnec.to_json_ntv(ndarray, **option)
    else:
        jsn, nam, typ = XndarrayConnec.to_json_ntv(ndarray, **option)
    name = nam if nam else ''
    return NpUtil.json_ntv(name, typ, jsn, header=option['header'], 
                           encoded=option['encoded'])

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

    '''NTV connector for Numpy ndarray.'''

    clas_obj = 'ndarray'
    clas_typ = 'ndarray'

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs):
        ''' convert json ntv_value into a Ndarray.
        
        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with 
        non Numpy ntv_type into data with python type'''
        #return Ndarray.read_json(ntv_value, **kwargs)
        return Ndarray.read_json2(ntv_value, **kwargs)

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        ''' convert a Ndarray (value, name, type) into NTV json (json-value, name, type).

        *Parameters*

        - **typ** : string (default None) - ntv_type of the ndarray object,
        - **name** : string (default None) - name of the ndarray object
        - **value** : ndarray value
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **format** : string (default 'full') - representation format of the ndarray,
        - **encoded** : Boolean (default False) - json-value if False else json-text
        - **header** : Boolean (default True) - including ndarray type
        '''
        option = {'format': 'full', 'header': True, 'encoded': False,
                  'notype': False, 'noshape': True, 'novalue': False} | kwargs
        if not option['format'] in ['full', 'complete']: 
            option['noshape'] = False
        return (Ndarray(value).to_json2(**option), name, 'ndarray')

class XndarrayConnec(NtvConnector):

    '''NTV connector for xndarray.'''

    clas_obj = 'Xndarray'
    clas_typ = 'xndarray'

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs): 
        ''' convert json ntv_value into a Xndarray.

        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with 
        non Numpy ntv_type into Xndarray with python type
        '''
        return Xndarray.read_json(ntv_value, **kwargs)

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        ''' convert a Xndarray (value, name, type) into NTV json (json-value, name, type).

        *Parameters*

        - **typ** : string (default None) - type of the NTV object,
        - **name** : string (default None) - name of the NTV object
        - **value** : ndarray values
        - **encoded** : Boolean (default False) - json-value if False else json-text
        - **header** : Boolean (default True) - including xndarray type
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **format** : string (default 'full') - representation format of the ndarray,
        - **extension** : string (default None) - type extension
        '''            
        option = {'notype': False, 'extension': None, 'format': 'full', 
                  'noshape': True, 'header': True, 'encoded': False,
                  'novalue': False} | kwargs
        if not option['format'] in ['full', 'complete']: 
            option['noshape'] = False
        option['noname'] = True
        return (value.to_json(**option), value.full_name, 'xndarray')

class XdatasetConnec(NtvConnector):

    '''NTV connector for xdataset.'''

    clas_obj = 'Xdataset'
    clas_typ = 'xdataset'

    @staticmethod
    def to_obj_ntv(ntv_value, **kwargs):  # reindex=True, decode_str=False):
        ''' convert json ntv_value into a Xndarray.

        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with 
        non Numpy ntv_type into Xndarray with python type
        '''
        return Xdataset.read_json(ntv_value, **kwargs)

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        ''' convert a Xdataset (value, name, type) into NTV json (json-value, name, type).

        *Parameters*

        - **typ** : string (default None) - not used,
        - **name** : string (default None) - not used
        - **value** : Xdataset entity
        - **encoded** : Boolean (default False) - json value if False else json text
        - **header** : Boolean (default True) - including 'xdataset' type
        - **notype** : list of Boolean (default list of None) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default False) - if True, without shape if dim < 1
        - **format** : list of string (default list of 'full') - representation format of the ndarray,
        '''            
        option = {'notype': False, 'extension': None, 'format': 'full', 
                  'noshape': True, 'header': True, 'encoded': False,
                  'novalue': False} | kwargs
        if not option['format'] in ['full', 'complete']: 
            option['noshape'] = False
        option['noname'] = True
        return (value.to_json(**option), value.name, 'xdataset')
