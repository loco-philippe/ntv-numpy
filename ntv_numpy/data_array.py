# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `data_array` module is part of the `ntv-numpy.ntv_numpy` package ([specification document](
https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)).

It contains the classes `Darray` (abstract), `Dfull`, `Dcomplete` for the
representation of unidimensional arrays.

For more information, see the
[user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html)
 or the [github repository](https://github.com/loco-philippe/ntv-numpy).

"""

#import datetime
#from decimal import Decimal

from abc import ABC, abstractmethod
import json
import numpy as np
from json_ntv import Ntv, NtvConnector
#from json_ntv import Ntv, ShapelyConnec, NtvConnector
import pandas as pd


class Darray(ABC):
    ''' The Darray class is an abstract class used by `Dfull`and `Dcomplete` classes.

    *Attributes :*
    - **data** :  np.ndarray - data after coding
    - **ref**:  int or string - reference to another Darray data
    - **coding**: np.ndarray of int - mapping between data and the values

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    '''

    def __init__(self, data, ref=None, coding=None, dtype=None, unidim=False):
        '''Darray constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (after coding)
        - **ref** : String or integer (default None) - name or index of another Darray
        - **coding**: List of integer (default None) - mapping between data and the list of values
        - **dtype**: string (default None) - numpy.dtype to apply
        '''
        if isinstance(data, Darray):
            self.data = data.data
            self.ref = data.ref
            self.coding = data.coding
            return
        data = data if isinstance(data, (list, np.ndarray)) else [data]
        if (len(data) > 0 and isinstance(data[0], (list, np.ndarray))) or unidim:
            self.data = np.fromiter(data, dtype='object')
        else:
            self.data = np.array(data, dtype=dtype).reshape(-1)
        self.ref = ref
        self.coding = np.array(coding)

    def __repr__(self):
        '''return classname and number of value'''
        return self.__class__.__name__ + '[' + str(len(self)) + ']'

    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_json())

    def __eq__(self, other):
        ''' equal if values are equal'''
        return np.array_equal(self.values, other.values, equal_nan=False)

    def __len__(self):
        ''' len of values'''
        return self._len_val

    def __contains__(self, item):
        ''' item of values'''
        return item in self.values

    def __getitem__(self, ind):
        ''' return value item'''
        if isinstance(ind, tuple):
            return [self.values[i] for i in ind]
            # return [copy(self.values[i]) for i in ind]
        return self.values[ind]
        # return copy(self.values[ind])

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)

    @staticmethod
    def read_json(val, dtype=None, unidim=False):
        ''' return a Darray entity from a list of data.

        *Parameters*

        - **val**: list of data
        - **dtype** : string (default None) - numpy.dtype to apply
        '''
        val = val if isinstance(val, list) else [val]
        if not val or not isinstance(val[0], list):
            return Dfull(val, dtype=dtype, unidim=unidim)
        match val:
            case [data, ref, list(coding)] if (isinstance(ref, (int, str)) and
                                               isinstance(coding[0], int) and
                                               max(coding) < len(data)):
                return None
            case [data, ref] if (isinstance(data, list) and
                                 isinstance(ref, (int, str))):
                return None
            case [data, list(coef)] if len(coef) == 1:
                return None
            case [data, list(coding)] if (isinstance(coding[0], int) and
                                          max(coding) < len(data)):
                return Dcomplete(data, None, coding, dtype=dtype, unidim=unidim)
            case _:
                return Dfull(val, dtype=dtype, unidim=unidim)

    @abstractmethod
    def to_json(self):
        ''' return a JsonValue'''

    @property
    @abstractmethod
    def values(self):
        ''' return the list of values'''

    @property
    @abstractmethod
    def _len_val(self):
        '''return the length of the entity'''


class Dfull(Darray):
    ''' Representation of a one dimensional Array with full representation

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    '''

    def __init__(self, data, ref=None, coding=None, dtype=None, unidim=False):
        '''Dfull constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (after coding)
        - **ref** : unused
        - **coding**: unused
        - **dtype**: string (default None) - numpy.dtype to apply
        '''
        super().__init__(data, dtype=dtype, unidim=unidim)

    def to_json(self):
        ''' return a JsonValue of the Dfull entity.'''
        #return self.data.tolist()
        return Dutil.list_json(self.data)

    @property
    def values(self):
        ''' return the list of values'''
        return self.data

    @property
    def _len_val(self):
        '''return the length of the Dfull entity'''
        return len(self.data) if self.data.ndim > 0 else 0


class Dcomplete(Darray):
    ''' Representation of a one dimensional Array with full representation

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    '''

    def __init__(self, data, ref=None, coding=None, dtype=None, unidim=False):
        '''Dcomplete constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (after coding)
        - **ref** : unused
        - **coding**: List of integer (default None) - mapping between data and the list of values
        - **dtype**: string (default None) - numpy.dtype to apply
        '''
        if coding is None:
            try:
                data, coding = np.unique(data, return_inverse=True)
            except (TypeError, ValueError):
                dat, idx, coding = np.unique(np.frompyfunc(Ntv.from_obj, 1, 1)(data),
                                             return_index=True, return_inverse=True)
                data = data[idx]
        super().__init__(data, coding=coding, dtype=dtype, unidim=unidim)

    def to_json(self):
        ''' return a JsonValue of the Dcomplete entity.'''
        #return [self.data.tolist(), self.coding.tolist()]
        return [Dutil.list_json(self.data), self.coding.tolist()]

    @property
    def values(self):
        ''' return the list of values'''
        return self.data[self.coding]

    @property
    def _len_val(self):
        '''return the length of the Dcomplete entity'''
        return len(self.coding) if self.coding.ndim > 0 else 0

class Dutil:
    '''np.ndarray utilities.

    *static methods*
    - `convert`
    - `is_json`
    - `ntv_val`
    - `add_ext`
    - `split_type`
    - `ntv_type`
    - `nda_ntv_type`
    - `dtype`
    - `json_ntv`
    - `split_name`
    - `split_json_name`
    '''
    """DATATION_DT = {'date': 'datetime64[D]', 'year': 'datetime64[Y]',
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
    
    DT_NTVTYPE = DT_DATATION | DT_LOCATION | DT_OTHER | DT_CONNECTOR | DT_PYTHON"""
    
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
            equal = {np.ndarray: Dutil.equals,
                     pd.Series: SeriesConnec.equals,
                     pd.DataFrame: DataFrameConnec.equals}
            for nps, npo in zip(nself, nother):
                if not equal[type(nself[0])](nps, npo):
                    return False
            return True
        return np.array_equal(nself, nother)

    @staticmethod 
    def list_json(nda):
        '''return a JSON representation of a unidimensional np.ndarray'''
        if len(nda) == 0:
            return []
        if isinstance(nda[0], np.ndarray):
            return [Dutil.list_json(arr) for arr in nda]
        return nda.tolist()    
    
    """@staticmethod
    def convert(ntv_type, nda, tojson=True, convert=True):
        ''' convert np.ndarray with external NTVtype.

        *Parameters*

        - **ntv_type** : string - NTVtype deduced from the np.ndarray name_type and dtype,
        - **nda** : np.ndarray to be converted.
        - **tojson** : boolean (default True) - apply to json function
        - **convert** : boolean (default True) - If True, convert json data with
        non Numpy ntv_type into data with python type
        '''
        if tojson:
            match ntv_type:
                case dat if dat in Dutil.DATATION_DT:
                    return nda.astype(Dutil.DATATION_DT[dat]).astype(str)
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
                case [dat, _] if dat in Dutil.DATATION_DT:
                    return nda.astype(Dutil.DATATION_DT[dat])
                case [std, _] if std in Dutil.OTHER_DT:
                    return nda.astype(Dutil.OTHER_DT[std])
                case ['time', True]:
                    return np.frompyfunc(datetime.time.fromisoformat, 1, 1)(nda)
                case ['decimal64', True]:
                    return np.frompyfunc(Decimal, 1, 1)(nda)
                case ['narray', True]:
                    nar = np.frompyfunc(Ndarray.read_json, 1, 1)(nda)
                    return np.frompyfunc(Ndarray.to_ndarray, 1, 1)(nar)
                case ['ndarray', True]:
                    return np.frompyfunc(Ndarray.read_json, 1, 1)(nda)
                case [python, _] if python in Dutil.PYTHON_DT:
                    return nda.astype('object')
                case [connec, True] if connec in Dutil.CONNECTOR_DT:
                    return np.fromiter([NtvConnector.uncast(nd, None, connec)[0]
                                        for nd in nda], dtype='object')
                case [('point' | 'line' | 'polygon' | 'geometry'), True]:
                    return np.frompyfunc(ShapelyConnec.to_geometry, 1, 1)(nda)
                case [_, False]:
                    return nda.astype(Dutil.CONVERT_DT[
                        Dutil.dtype(ntv_type, convert=False)])
                case _:
                    return nda.astype(Dutil.dtype(ntv_type))

        # float.fromhex(x.hex()) == x, bytes(bytearray.fromhex(x.hex())) == x"""
    
    
    
    