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

from abc import ABC, abstractmethod
import json
import numpy as np
import pandas as pd
from json_ntv import Ntv, NtvConnector


class Darray(ABC):
    """The Darray class is an abstract class used by `Dfull`and `Dcomplete` classes.

    *Attributes :*
    - **data** :  np.ndarray - data after coding
    - **ref**:  int or string - reference to another Darray data
    - **coding**: np.ndarray of int - mapping between data and the values

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    """

    #def __init__(self, data, ref=None, coding=None, dtype=None, unidim=False):
    def __init__(self, data, **kwargs):
        """Darray constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (after coding)
        - **ref** : String or integer (default None) - name or index of another Darray
        - **coding**: List of integer (default None) - mapping between data and the list of values
        - **dtype**: string (default None) - numpy.dtype to apply
        """
        option = {'ref': None, 'coding': None, 'dtype': None, 'unidim': False} | kwargs
        if isinstance(data, Darray):
            self.data = data.data
            self.ref = data.ref
            self.coding = data.coding
            return
        data = data if isinstance(data, (list, np.ndarray)) else [data]
        if (len(data) > 0 and isinstance(data[0], (list, np.ndarray))) or option['unidim']:
            dtype = data.dtype if isinstance(data, np.ndarray) else "object"
            self.data = np.fromiter(data, dtype=dtype)
        else:
            self.data = np.array(data, dtype=option['dtype']).reshape(-1)
        self.ref = option['ref']
        self.coding = np.array(option['coding'])
        return

    def __repr__(self):
        """return classname and number of value"""
        return self.__class__.__name__ + "[" + str(len(self)) + "]"

    def __str__(self):
        """return json string format"""
        return json.dumps(self.to_json())

    def __eq__(self, other):
        """equal if values are equal"""
        return np.array_equal(self.values, other.values, equal_nan=False)

    def __len__(self):
        """len of values"""
        return self._len_val

    def __contains__(self, item):
        """item of values"""
        return item in self.values

    def __getitem__(self, ind):
        """return value item"""
        if isinstance(ind, tuple):
            return [self.values[i] for i in ind]
            # return [copy(self.values[i]) for i in ind]
        return self.values[ind]
        # return copy(self.values[ind])

    def __copy__(self):
        """Copy all the data"""
        return self.__class__(self)

    @staticmethod 
    def decode_json(jsn):
        """return a dict of parameters deduced from jsn

        *return*: dict
  
        - **uri**: string
        - **data**: list of values
        - **keys**: list of integers
        - **leng**: integer
        - **coef**: integer
        - **sp_idx**: list of integers
        - **custom**: dict
        """
        uri = data = keys = leng = coef = sp_idx = custom = None
        match jsn:
            case str(uri):...
            case [list(data), coding]:
                match coding: 
                    case [int(leng), code]:
                        match code: 
                            case int(coef):...
                            case [list(keys), list(sp_idx)]:...
                            case list(sp_idx):...
                            case _:...
                    case list(keys):...
                    case dict(custom):...
                    case _: ...
            case list(data):...
            case _: ...
        return {'uri': uri, 'data': data, 'keys': keys, 'leng': leng, 
                  'coef': coef, 'sp_idx': sp_idx, 'custom': custom}

    @staticmethod
    def read_json(val, dtype=None, unidim=False):
        """return a Darray entity from a list of data.

        *Parameters*

        - **val**: list of data
        - **dtype** : string (default None) - numpy.dtype to apply
        """
        val = val if isinstance(val, list) else [val]
        if not val or not isinstance(val[0], list):
            return Dfull(val, dtype=dtype, unidim=unidim)
        match val:
            case [data, ref, list(coding)] if (
                isinstance(ref, (int, str))
                and isinstance(coding[0], int)
                and max(coding) < len(data)
            ):
                return None
            case [data, ref] if (
                isinstance(data, list) and isinstance(ref, (int, str))
            ):
                return None
            case [data, list(coef)] if len(coef) == 1:
                return None
            case [data, list(coding)] if (
                isinstance(coding[0], int) and max(coding) < len(data)
            ):
                return Dcomplete(data, coding=coding, dtype=dtype, unidim=unidim)
            case _:
                return Dfull(val, dtype=dtype, unidim=unidim)

    @abstractmethod
    def to_json(self):
        """return a JsonValue"""

    @property
    @abstractmethod
    def values(self):
        """return the list of values"""

    @property
    @abstractmethod
    def _len_val(self):
        """return the length of the entity"""


class Dfull(Darray):
    """Representation of a one dimensional Array with full representation

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    """

    def __init__(self, data, **kwargs):
        """Dfull constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (after coding)
        - **dtype**: string (default None) - numpy.dtype to apply
        """
        option = {'dtype': None, 'unidim': False} | kwargs
        super().__init__(data, **option)

    def to_json(self):
        """return a JsonValue of the Dfull entity."""
        return Dutil.list_json(self.data)

    @property
    def values(self):
        """return the list of values"""
        return self.data

    @property
    def _len_val(self):
        """return the length of the Dfull entity"""
        return len(self.data) if self.data.ndim > 0 else 0


class Dcomplete(Darray):
    """Representation of a one dimensional Array with full representation

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    """

    def __init__(self, data, **kwargs):
        """Dcomplete constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (after coding)
        - **coding**: List of integer (default None) - mapping between data and the list of values
        - **dtype**: string (default None) - numpy.dtype to apply
        """
        option = {'coding': None, 'dtype': None, 'unidim': False} | kwargs
        coding = option['coding']
        if coding is None:
            try:
                data, coding = np.unique(data, return_inverse=True)
            except (TypeError, ValueError):
                dat, idx, coding = np.unique(
                    np.frompyfunc(Ntv.from_obj, 1, 1)(data),
                    return_index=True,
                    return_inverse=True,
                )
                data = data[idx]
        super().__init__(data, coding=coding, dtype=option['dtype'], unidim=option['unidim'])

    def to_json(self):
        """return a JsonValue of the Dcomplete entity."""
        return [Dutil.list_json(self.data), self.coding.tolist()]

    @property
    def values(self):
        """return the list of values"""
        return self.data[self.coding]

    @property
    def _len_val(self):
        """return the length of the Dcomplete entity"""
        return len(self.coding) if self.coding.ndim > 0 else 0

'''class Dsparse(Darray):
    """Representation of a one dimensional Array with sparse representation

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    """

    def __init__(self, data, **kwargs):
        """Dsparse constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (sp_value + fill_value)
        - **coding**: List of integer (default None) - mapping between data and values (sp_index)
        - **dtype**: string (default None) - numpy.dtype to apply
        """
        option = {'sp_value': None, 'sp_index': None, 'fill_value': None, 
                  'length': None, 'dtype': None, 'unidim': False} | kwargs
        length, idx = option['coding']
        coding = coding if is None:
            try:
                data, coding = np.unique(data, return_inverse=True)
            except (TypeError, ValueError):
                dat, idx, coding = np.unique(
                    np.frompyfunc(Ntv.from_obj, 1, 1)(data),
                    return_index=True,
                    return_inverse=True,
                )
                data = data[idx]
        super().__init__(data, coding=coding, dtype=option['dtype'], unidim=option['unidim'])

    def to_json(self):
        """return a JsonValue of the Dcomplete entity."""
        return [Dutil.list_json(self.data), self.coding.tolist()]

    @property
    def values(self):
        """return the list of values"""
        return self.data[self.coding]

    @property
    def _len_val(self):
        """return the length of the Dcomplete entity"""
        return len(self.coding) if self.coding.ndim > 0 else 0
'''
class Dutil:
    """np.ndarray utilities.

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
    """

    @staticmethod
    def equals(nself, nother):
        """return True if all elements are equals and dtype are equal"""
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
            SeriesConnec = NtvConnector.connector().get("SeriesConnec")
            DataFrameConnec = NtvConnector.connector().get("DataFrameConnec")
            equal = {
                np.ndarray: Dutil.equals,
                pd.Series: SeriesConnec.equals,
                pd.DataFrame: DataFrameConnec.equals,
            }
            for nps, npo in zip(nself, nother):
                if not equal[type(nself[0])](nps, npo):
                    return False
            return True
        return np.array_equal(nself, nother)

    @staticmethod
    def list_json(nda):
        """return a JSON representation of a unidimensional np.ndarray"""
        if len(nda) == 0:
            return []
        if isinstance(nda[0], np.ndarray):
            return [Dutil.list_json(arr) for arr in nda]
        return nda.tolist()
