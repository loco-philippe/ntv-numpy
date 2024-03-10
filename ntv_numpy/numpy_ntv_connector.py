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

import json
import numpy as np
from ntv_numpy.ndarray import Ndarray, NpUtil

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
    - **noshape** : Boolean (default True) - if False, only shape if dim > 1
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
    """if option['header'] or name:
        typ = ':' + typ if option['header'] else ''
        jsn = {name + typ : jsn}
    if option['encoded']:
        return json.dumps(jsn)
    return jsn"""

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
        return Ndarray.read_json(ntv_value, **kwargs)

    @staticmethod
    def to_json_ntv(value, name=None, typ=None, **kwargs):
        ''' convert a ndarray (value, name, type) into NTV json (json-value, name, type).

        *Parameters*

        - **typ** : string (default None) - ntv_type of the ndarray object,
        - **name** : string (default None) - name of the ndarray object
        - **value** : ndarray value
        - **noshape** : Boolean (default True) - if False, only shape if dim > 1
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **format** : string (default 'full') - representation format of the ndarray,
        - **extension** : string (default None) - type extension
        '''
        option = {'notype': False, 'extension': None, 'format': 'full',
                  'noshape': True, 'novalue': False} | kwargs
        return (Ndarray.to_json(value, ntv_type=typ, **option), name, 'ndarray')

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
