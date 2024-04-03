# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `ndarray` module is part of the `ntv-numpy.ntv_numpy` package 
([specification document](
https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)).

It contains the classes `Ndarray`, `NpUtil`, `NdarrayError` for the JSON interface
of numpy.ndarrays.

For more information, see the
[user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html)
 or the [github repository](https://github.com/loco-philippe/ntv-numpy).

"""

import datetime
import json

from decimal import Decimal
import pandas as pd
import numpy as np
from json_ntv import Ntv, ShapelyConnec, Datatype, NtvConnector
from ntv_numpy.data_array import Dfull, Dcomplete, Darray


class Ndarray:
    ''' The Ndarray class is the JSON interface of numpy.ndarrays.

    *static methods*
    - `read_json`
    - `to_json`
    - `equals`
    '''

    def __init__(self, dar, ntv_type=None, shape=None):
        '''Ndarray constructor.

        *Parameters*

        - **dar**: Darray or np.ndarray - data to represent
        - **shape** : list of integer (default None) - length of dimensions
        - **ntv_type**: string (default None) - NTVtype to apply
        '''
        dar = None if isinstance(dar, list) and len(dar) == 0 else dar
        if isinstance(dar, Ndarray):
            self.uri = dar.uri
            self.is_json = dar.is_json
            self.ntvtype = dar.ntvtype
            self.shape = dar.shape
            self.darray = dar.darray
            return
        if isinstance(dar, str):
            self.uri = dar
            self.is_json = True
            self.ntvtype = Datatype(ntv_type) if ntv_type else None
            self.shape = shape
            self.darray = None
            return
        if shape:
            dar = Dfull(dar, dtype=NpUtil.dtype(ntv_type), unidim=True).data
        else:
            dar = np.array(dar if isinstance(dar, (list, np.ndarray))
                              else [dar], dtype=NpUtil.dtype(ntv_type))
            shape = list(dar.shape)
        dar = np.array(dar).reshape(-1)
        ntv_type = NpUtil.nda_ntv_type(dar) if not (ntv_type or dar is None) else ntv_type
        self.uri = None
        self.is_json = NpUtil.is_json(dar[0])
        self.ntvtype = Datatype(ntv_type)
        self.shape = shape
        self.darray = dar.astype(NpUtil.dtype(str(self.ntvtype)))

    def __repr__(self):
        '''return classname, the shape and the ntv_type'''
        return self.__class__.__name__ + '(' + self.ntv_type + ', ' + str(self.shape) + ')'

    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_json())

    def __eq__(self, other):
        ''' equal if attributes are equal'''
        if self.ntv_type != other.ntv_type:
            return False
        if self.uri != other.uri:
            return False
        if self.shape != other.shape:
            return False
        if self.darray is None and other.darray is None:
            return True
        if self.darray is None or other.darray is None:
            return False
        return Ndarray.equals(self.darray, other.darray)

    def __len__(self):
        ''' len of ndarray'''
        #return len(self.darray) if self.darray is not None else Ndarray.len_shape(self.shape)
        return len(self.darray) if self.darray is not None else 0

    def __contains__(self, item):
        ''' item of darray values'''
        return item in self.darray if self.darray is not None else None

    def __getitem__(self, ind):
        ''' return darray value item'''
        if self.darray is None:
            return None
        if isinstance(ind, tuple):
            return [self.darray[i] for i in ind]
        return self.darray[ind]

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)

    @property
    def ntv_type(self):
        ''' string representation of ntvtype'''
        return str(self.ntvtype) if self.ntvtype else None

    @property 
    def ndarray(self):
        '''representation with a np.ndarray not flattened'''
        return self.darray.reshape(self.shape) if not self.darray is None else None

    def set_array(self, darray):
        '''set a new darray and return the removed uri link'''
        ntv_type = None
        shape = None
        if isinstance(darray, Ndarray):
            darray = darray.darray
            ntv_type = darray.ntv_type
            shape = darray.shape
        darray = np.array(darray).reshape(-1)
        self.shape = shape if self.shape is None else self.shape
        self.ntv_type = ntv_type if self.ntv_type is None else self.ntv_type
        if len(darray) != Ndarray.len_shape(shape):
            return
        self.uri = None
        self.darray = darray
        return
    
    def set_uri(self, uri):
        '''set a new uri and return the removed np.ndarray'''


    def to_ndarray(self):
        '''representation with a np.ndarray not flattened'''
        return self.ndarray
    
    @property
    def mode(self):
        '''representation mode of the darray/uri data (relative, absolute,
        undefined, inconsistent)'''
        match [self.darray, self.uri]:
            case [None, str()]:
                return 'relative'
            case [None, None]:
                return 'undefined'
            case [_, None]:
                return 'absolute'
            case _:
                return 'inconsistent'

    @staticmethod
    def read_json(jsn, **kwargs):
        ''' convert json ntv_value into a ndarray.


        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with
        non Numpy ntv_type into data with python type
        '''
        option = {'convert': True} | kwargs
        jso = json.loads(jsn) if isinstance(jsn, str) else jsn
        ntv_value, = Ntv.decode_json(jso)[:1]

        ntv_type = None
        shape = None
        match ntv_value[:-1]:
            case []: ...
            case [ntv_type, shape]: ...
            case [str(ntv_type)]: ...
            case [list(shape)]: ...
        unidim = not shape is None
        if isinstance(ntv_value[-1], str):
            return Ndarray(ntv_value[-1], shape=shape, ntv_type=ntv_type)
        darray = Darray.read_json(ntv_value[-1], dtype=NpUtil.dtype(ntv_type), 
                                  unidim=unidim)
        darray.data = NpUtil.convert(ntv_type, darray.data, tojson=False,
                                     convert=option['convert'])
        return Ndarray(darray.values, shape=shape, ntv_type=ntv_type)

    def to_json(self, **kwargs):
        ''' convert a Ndarray into json-value

        *Parameters*

        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **format** : string (default 'full') - representation format of the ndarray,
        - **encoded** : Boolean (default False) - json-value if False else json-text
        - **header** : Boolean (default True) - including ndarray type
        '''
        option = {'format': 'full', 'header': True, 'encoded': False,
                  'notype': False, 'noshape': True, 'novalue': False} | kwargs
        if self.mode in ['undefined', 'inconsistent']:
            return None
        if self.mode == 'absolute' and len(self.darray) == 0:
            return [[]]

        shape = None if not self.shape or (len(self.shape) < 2 and 
                                           option['noshape']) else self.shape

        if self.mode == 'relative':
            js_val = self.uri
        else:
            js_val = NpUtil.ntv_val(self.ntv_type, self.darray, option['format'],
                                    self.is_json) if not option['novalue'] else ['-']

        lis = [self.ntv_type if not option['notype'] else None, shape, js_val]
        return NpUtil.json_ntv(None, 'ndarray',
                               [val for val in lis if not val is None],
                               header=option['header'], encoded=option['encoded'])

    @staticmethod
    def equals(nself, nother):
        '''return True if all elements are equals and dtype are equal'''
        if not (isinstance(nself, np.ndarray) and isinstance(nother, np.ndarray)):
            return False
        if nself.dtype != nother.dtype or nself.shape != nother.shape:
            return False
        if len(nself.shape) == 0:
            return True
        if len(nself) != len(nother):
            return False
        if len(nself) == 0:
            return True
        if isinstance(nself[0], (np.ndarray, pd.Series, pd.DataFrame)):
            SeriesConnec = NtvConnector.connector().get('SeriesConnec')
            DataFrameConnec = NtvConnector.connector().get('DataFrameConnec')
            equal = {np.ndarray: Ndarray.equals,
                     pd.Series: SeriesConnec.equals,
                     pd.DataFrame: DataFrameConnec.equals}
            for nps, npo in zip(nself, nother):
                if not equal[type(nself[0])](nps, npo):
                    return False
            return True
        return np.array_equal(nself, nother)

    @property
    def info(self):
        ''' infos of the Ndarray'''
        inf = {'shape': self.shape}
        inf['length'] = len(self)
        inf['ntvtype'] = self.ntv_type
        inf['shape'] = self.shape
        inf['uri'] = self.uri
        return {key: val for key, val in inf.items() if val}

    @staticmethod
    def len_shape(shape):
        prod = 1
        for dim in shape:
            prod *= dim
        return prod


class NpUtil:
    '''ntv-ndarray utilities.

    *static methods*
    - `convert`
    - `ntv_val`
    - `add_ext`
    - `split_type`
    - `ntv_type`
    - `nda_ntv_type`
    - `dtype`
    - `json_ntv`
    '''

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

    #CONNECTOR_DT = {'field': 'Series', 'tab': 'DataFrame'}
    CONNECTOR_DT = {'field': 'Series', 'tab': 'DataFrame'}
    DT_CONNECTOR = {val: key for key, val in CONNECTOR_DT.items()}

    PYTHON_DT = {'array': 'list', 'time': 'datetime.time',
                 'object': 'dict', 'null': 'NoneType', 'decimal64': 'Decimal',
                 'ndarray': 'ndarray', 'narray': 'narray'}
    DT_PYTHON = {val: key for key, val in PYTHON_DT.items()}

    #OTHER_DT = {'boolean': 'bool', 'string': 'str'}
    OTHER_DT = {'boolean': 'bool', 'string': 'str', 'base16': 'bytes'}
    DT_OTHER = {val: key for key, val in OTHER_DT.items()}

    LOCATION_DT = {'point': 'Point',
                   'line': 'LineString', 'polygon': 'Polygon'}
    DT_LOCATION = {val: key for key, val in LOCATION_DT.items()}

    NUMBER_DT = {'json': 'object', 'number': None, 'month': 'int', 'day': 'int',
                 'wday': 'int', 'yday': 'int', 'week': 'hour', 'minute': 'int',
                 'second': 'int'}
    #STRING_DT = {'base16': 'str', 'base32': 'str', 'base64': 'str',
    STRING_DT = {'base32': 'str', 'base64': 'str',
                 'period': 'str', 'duration': 'str', 'jpointer': 'str',
                 'uri': 'str', 'uriref': 'str', 'iri': 'str', 'iriref': 'str',
                 'email': 'str', 'regex': 'str', 'hostname': 'str', 'ipv4': 'str',
                 'ipv6': 'str', 'file': 'str', 'geojson': 'str', }
    FORMAT_CLS = {'full': Dfull, 'complete': Dcomplete}
    CONVERT_DT = {'object': 'object', 'array': 'object', 'json': 'object',
                  'number': 'float', 'boolean': 'bool', 'null': 'object',
                  'string': 'str', 'integer': 'int'}
    STRUCT_DT = {'Ntv': 'object', 'NtvSingle': 'object', 'NtvList': 'object'}
    
    DT_NTVTYPE = DT_DATATION | DT_LOCATION | DT_OTHER | DT_CONNECTOR | DT_PYTHON

    @staticmethod
    def is_json(obj):
        ''' check if obj is a json structure and return True if obj is a json-value

        *Parameters*

        - **obj** : object to check'''
        if obj is None:
            return True
        is_js = NtvConnector.is_json
        match obj:
            case str() | int() | float() | bool():
                return True
            case list() | tuple() as obj:
                if not obj:
                    return True
                return min(is_js(obj_in) for obj_in in obj)
            case dict() as obj:
                if not obj:
                    return True
                if not min(isinstance(key, str) for key in obj.keys()):
                    return False
                return min(is_js(obj_in) for obj_in in obj.values())
            case _:
                return False

    @staticmethod
    def convert(ntv_type, nda, tojson=True, convert=True):
        ''' convert ndarray with external NTVtype.

        *Parameters*

        - **ntv_type** : string - NTVtype deduced from the ndarray name_type and dtype,
        - **nda** : ndarray to be converted.
        - **tojson** : boolean (default True) - apply to json function
        - **convert** : boolean (default True) - If True, convert json data with
        non Numpy ntv_type into data with python type
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
            match [ntv_type, convert]:
                case [None, _]:
                    return nda
                case [dat, _] if dat in NpUtil.DATATION_DT:
                    return nda.astype(NpUtil.DATATION_DT[dat])
                case [std, _] if std in NpUtil.OTHER_DT:
                    return nda.astype(NpUtil.OTHER_DT[std])
                case ['time', True]:
                    return np.frompyfunc(datetime.time.fromisoformat, 1, 1)(nda)
                case ['decimal64', True]:
                    return np.frompyfunc(Decimal, 1, 1)(nda)
                case ['narray', True]:
                    nar = np.frompyfunc(Ndarray.read_json, 1, 1)(nda)
                    return np.frompyfunc(Ndarray.to_ndarray, 1, 1)(nar)
                case ['ndarray', True]:
                    return np.frompyfunc(Ndarray.read_json, 1, 1)(nda)
                case [python, _] if python in NpUtil.PYTHON_DT:
                    return nda.astype('object')
                case [connec, True] if connec in NpUtil.CONNECTOR_DT:
                    return np.fromiter([NtvConnector.uncast(nd, None, connec)[0]
                                        for nd in nda], dtype='object')
                case [('point' | 'line' | 'polygon' | 'geometry'), True]:
                    return np.frompyfunc(ShapelyConnec.to_geometry, 1, 1)(nda)
                case [_, False]:
                    return nda.astype(NpUtil.CONVERT_DT[
                        NpUtil.dtype(ntv_type, convert=False)])
                case _:
                    return nda.astype(NpUtil.dtype(ntv_type))

    @staticmethod
    def ntv_val(ntv_type, nda, form, is_json=False):
        ''' convert a np.ndarray into NTV json-value.

        *Parameters*

        - **ntv_type** : string - NTVtype deduced from the ndarray, name_type and dtype,
        - **nda** : ndarray to be converted.
        - **form** : format of data ('full', 'complete', 'sparse', 'primary').
        - **is_json** : boolean (defaut False) - True if nda data is Json data
        '''
        if form == 'complete' and len(nda) < 2:
            raise NdarrayError(
                "complete format is not available with ndarray length < 2")
        Format = NpUtil.FORMAT_CLS[form]
        darray = Format(nda)
        ref = darray.ref
        coding = darray.coding
        if is_json:
            return Format(darray.data, ref=ref, coding=coding).to_json()
        match ntv_type:
            case 'narray':
                data = [Ndarray(nd).to_json(header=False) for nd in darray.data]
            case 'ndarray':
                #data = [Ndarray.to_json(nd) for nd in darray.data]
                data = [Ndarray(nd).to_json(header=False) for nd in darray.data]
            case connec if connec in NpUtil.CONNECTOR_DT:
                data = [NtvConnector.cast(nd, None, connec)[0]
                        for nd in darray.data]
            case 'point' | 'line' | 'polygon' | 'geometry':
                data = np.frompyfunc(ShapelyConnec.to_coord, 1, 1)(darray.data)
            case None:
                data = nda
            case _:
                data = NpUtil.convert(ntv_type, darray.data)
        return Format(data, ref=ref, coding=coding).to_json()

    @staticmethod
    def add_ext(typ, ext):
        '''return extended type : "typ[ext]"'''
        ext = '[' + ext + ']' if ext else ''
        return '' if not typ else typ + ext

    @staticmethod
    def split_type(typ):
        '''return a tuple with typ and extension'''
        if not isinstance(typ, str):
            return (None, None)
        spl = typ.split('[', maxsplit=1)
        return (spl[0], None) if len(spl) == 1 else (spl[0], spl[1][:-1])

    @staticmethod
    def split_json_name(string, notnone=False):
        '''return a tuple with name, ntv_type from string'''
        null = '' if notnone else None
        if not string or string == ':':
            return (null, null)
        spl = string.rsplit(':', maxsplit=1)
        if len(spl) == 1:
            return (string, null)
        if spl[0] == '':
            return (null, spl[1])
        sp0 = spl[0][:-1] if spl[0][-1] == ':' else spl[0]
        return (null if sp0 == '' else sp0, null if spl[1] == '' else spl[1])

    @staticmethod
    def ntv_type(dtype, ntv_type=None, ext=None):
        ''' return NTVtype from dtype, additional type and extension.

        *Parameters*

        - **dtype** : string - dtype of the ndarray
        - **ntv_type** : string - additional type
        - **ext** : string - type extension
        '''
        if ntv_type:
            return NpUtil.add_ext(ntv_type, ext)
        match dtype:
            case dat if dat in NpUtil.DT_NTVTYPE:
                return NpUtil.add_ext(NpUtil.DT_NTVTYPE[dat], ext)
            case string if string[:3] == 'str':
                return NpUtil.add_ext('string', ext)
            case byte if byte[:5] == 'bytes':
                #return NpUtil.add_ext('bytes', ext)
                return NpUtil.add_ext('base16', ext)
            case _:
                return NpUtil.add_ext(dtype, ext)

    @staticmethod
    def nda_ntv_type(nda, ntv_type=None, ext=None):
        '''return ntv_type from an ndarray, additional type and extension.

        *Parameters*

        - **nda** : ndarray - data used to calculate the ntv_type
        - **ntv_type** : string - additional type
        - **ext** : string - type extension
        '''
        dtype = nda.dtype.name
        pytype = nda.flat[0].__class__.__name__
        dtype = pytype if dtype == 'object' and not pytype in NpUtil.STRUCT_DT else dtype
        #dtype = pytype if dtype == 'object' and pytype in NpUtil.DT_NTVTYPE else dtype
        #dtype = nda.flat[0].__class__.__name__ if dtype == 'object' else dtype
        return NpUtil.ntv_type(dtype, ntv_type, ext)

    @staticmethod
    def dtype(ntv_type, convert=True):
        ''' return dtype from ntv_type

        *parameters*

        - **convert** : boolean (default True) - if True, dtype if from converted data
        '''
        DTYPE = (NpUtil.DATATION_DT | NpUtil.NUMBER_DT | NpUtil.OTHER_DT |
                 NpUtil.STRING_DT)
        OBJECT = NpUtil.LOCATION_DT | NpUtil.CONNECTOR_DT | NpUtil.PYTHON_DT
        type_base = NpUtil.split_type(ntv_type)[0]
        if convert:
            if type_base in OBJECT:
                return 'object'
            return DTYPE.get(ntv_type, DTYPE.get(type_base, type_base))
        return Datatype(ntv_type).json_type

    @staticmethod
    def json_ntv(ntv_name, ntv_type, ntv_value, **kwargs):
        ''' return the JSON representation of a NTV entity

        *parameters*

        - **ntv_name** : string - name of the NTV
        - **ntv_type** : string - type of the NTV
        - **ntv_value** : string - Json value of the NTV
        - **encoded** : boolean (default False) - if True return JsonText else JsonValue
        - **header** : boolean (default True) - if True include ntv_name + ntv_type
        '''
        name = ntv_name if ntv_name else ''
        option = {'encoded': False, 'header': True} | kwargs
        if option['header'] or name:
            typ = ':' + ntv_type if option['header'] and ntv_type else ''
            jsn = {name + typ: ntv_value} if name + typ else ntv_value
        else:
            jsn = ntv_value
        if option['encoded']:
            return json.dumps(jsn)
        return jsn


class NdarrayError(Exception):
    '''Multidimensional exception'''
