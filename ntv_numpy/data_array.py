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
    - **data** :  np.ndarray - data after coding (unidimensional)
    - **ref**:  list - list of parent keys
    - **coding**: list - parameters to cpnvert data to values
    - **keys**: np.ndarray of int - mapping between data and the values

    *dynamic values (@property)*
    - `values`

    *methods*
    - `decode_json` (staticmethod)
    - `read_json` (staticmethod)
    - `to_json`
    """

    def __init__(self, data, **kwargs):
        """Darray constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - unidimensional data to represent (after coding)
        - **ref** : String or integer (default None) - name or index of another Darray
        - **coding**: List of integer (default None) - mapping between data and the list of values
        - **dtype**: string (default None) - numpy.dtype to apply
        """
        option = {'ref': None, 'dtype': None} | kwargs
        if isinstance(data, Darray):
            self.data = data.data
            self.ref = data.ref
            self.coding = data.coding
            self.keys = data.keys
            self.codec = data.codec
            return
        data = data if isinstance(data, (list, np.ndarray)) else [data]
        dtype = data.dtype if isinstance(data, np.ndarray) else option['dtype']
        dtype = dtype if dtype else 'object'
        self.data = np.fromiter(data, dtype='object').astype(dtype)
        self.ref = option['ref']
        self.coding = None
        self.keys = None
        self.codec = self.data
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
        return len(self.keys) if self.keys.ndim > 0 else 0

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
            case str(uri): ...
            case [list(data), coding]:
                match coding:
                    case [int(val), list(code)] if val >= len(data):
                        leng = val
                        match code:
                            case [int(coef)]: ...
                            case [list(keys), list(sp_idx)]: ...
                            case list(sp_idx): ...
                            case _:
                                leng = data = None
                    case list(val) if isinstance(val[0], int) and max(val) < len(data):
                        keys = val
                    case dict(custom): ...
                    case _:
                        data = jsn
            case list(data): ...
            case _: ...
        return {'uri': uri, 'data': data, 'keys': keys, 'leng': leng,
                'coef': coef, 'sp_idx': sp_idx, 'custom': custom}

    @staticmethod
    def read_json(val, dtype=None, ref=None):
        """return a Darray entity from a list of data.

        *Parameters*

        - **val**: list of data
        - **dtype** : string (default None) - numpy.dtype to apply
        """
        params = Darray.decode_json(val) | {'ref': ref}
        list_params = [key for key, val in params.items() if val is not None]
        match list_params:
            case ['data']:
                return Dfull(params['data'], dtype=dtype)
            case ['data', 'keys']:
                return Dcomplete(params['data'], coding=params['keys'], dtype=dtype)
            case ['data', 'keys', 'ref']:
                return Drelative(params['data'], coding=params['keys'],
                                 ref=params['ref'], dtype=dtype)
            case ['data', 'leng', 'sp_idx']:
                return Dsparse(params['data'], dtype=dtype,
                               coding=[params['leng'], params['sp_idx']])
            case _:
                return
        return

    def to_json(self):
        """return a JsonValue"""
        if self.coding is not None:
            return [Dutil.list_json(self.data), self.coding]
        return Dutil.list_json(self.data)

    @property
    def values(self):
        """return the np.ndarray of values"""
        return self.codec[self.keys]


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
        option = {'dtype': None} | kwargs
        super().__init__(data, **option)
        self.keys = np.arange(len(self.data))

    @property
    def values(self):
        """return the np.ndarray of values"""
        return self.data


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

        - **data**: list, Darray or np.ndarray - data to represent (values or codec+coding)
        - **coding**: List of integer (default None) - mapping between data and the list of values
        - **dtype**: string (default None) - numpy.dtype to apply
        """
        option = {'coding': None, 'dtype': None} | kwargs
        super().__init__(data, **option)
        self.coding = self.coding if self.coding is not None else option['coding']
        if self.coding is not None:
            self.keys = np.array(self.coding)
            return
        try:
            values, coding = np.unique(self.data, return_inverse=True)
        except (TypeError, ValueError):
            idx, coding = np.unique(
                np.frompyfunc(Ntv.from_obj, 1, 1)(self.data),
                return_index=True,
                return_inverse=True,
            )[1:]
            values = self.data[idx]
        self.data = values
        self.coding = coding.tolist()
        self.keys = coding
        self.codec = self.data
        return

    @property
    def values(self):
        """return the np.ndarray of values"""
        return self.data[self.coding]
        # return self.codec[self.keys] #!!!


class Dsparse(Darray):
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

        - **data**: list, Darray or np.ndarray - data to represent (values or data+coding)
        - **coding**: List (default None) - sparse data coding (leng + sp_index)
        - **dtype**: string (default None) - numpy.dtype to apply
        """
        option = {'coding': None, 'dtype': None} | kwargs
        super().__init__(data, **option)
        self.coding = self.coding if self.coding is not None else option['coding']
        if self.coding is not None:
            self.keys = Dsparse._decoding(self.coding, np.arange(len(self.coding[1])))
            return
        leng = len(self.data)
        try:
            codec, cat, count = np.unique(self.data, return_inverse=True,
                                   return_counts=True)
        except (TypeError, ValueError):
            index, cat, count = np.unique(
                np.frompyfunc(Ntv.from_obj, 1, 1)(self.data),
                return_index=True,
                return_inverse=True,
                return_counts=True
            )[1:]
            codec = self.data[index]
        idx_fill = list(count).index(max(count))
        sp_index = [row for row, cat in zip(range(len(cat)), cat)
                    if cat != idx_fill] + [idx_fill]
        sp_values = self.data[sp_index]
        sp_index[-1] = -1
        self.coding = [leng, sp_index]
        self.data = sp_values
        self.keys = cat
        self.codec = codec
        return

    @property
    def values(self):
        """return the np.ndarray of values"""
        #return Dsparse._decoding(self.coding, self.data)
        return self.codec[self.keys]

    @staticmethod
    def _decoding(coding, data):
        """return values from coding and data"""
        leng, sp_index = coding
        try:
            values = np.full([leng], data[-1])
        except ValueError:
            values = np.fromiter([data[-1]] * leng, dtype='object')
        values[sp_index[:-1]] = data[:-1]
        return values


class Drelative(Darray):
    """Representation of a one dimensional Array with relative representation

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    """

    def __init__(self, data, **kwargs):
        """Drelative constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (values or data+coding)
        - **coding**: List (default None) - relative data coding (relative keys)
        - **dtype**: string (default None) - numpy.dtype to apply
        - **ref**: List (default None) - parent keys
        """
        option = {'coding': None, 'dtype': None, 'ref': None} | kwargs
        super().__init__(data, **option)
        self.coding = self.coding if self.coding is not None else option['coding']
        if self.coding is not None:
            self.keys = np.array(self.coding)[option['ref']]
            return
        self_dcomp = Dcomplete(self.data)
        derkeys = np.full([max(option['ref'])+1], -1)
        derkeys[option['ref']] = self_dcomp.keys
        if min(derkeys) < 0:
            raise DarrayError("parent is not a derive Field")
        self.data = self_dcomp.data
        self.coding = derkeys.tolist()
        self.keys = self_dcomp.keys
        self.codec = self.data
        return

    @property
    def values(self):
        """return the np.ndarray of values"""
        return self.codec[self.keys]

class Dimplicit(Darray):
    """Representation of a one dimensional Array with implicit representation

    *dynamic values (@property)*
    - `values`

    *methods*
    - `read_json` (staticmethod)
    - `to_json`
    """

    def __init__(self, data, **kwargs):
        """Drelative constructor.

        *Parameters*

        - **data**: list, Darray or np.ndarray - data to represent (values or data+coding)
        - **dtype**: string (default None) - numpy.dtype to apply
        - **ref**: List (default None) - parent keys
        """
        option = {'dtype': None, 'ref': None} | kwargs
        super().__init__(data, **option)
        self.keys = np.array(option['ref'])
        return

    @property
    def values(self):
        """return the np.ndarray of values"""
        return self.codec[self.keys]


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


class DarrayError(Exception):
    """Unidimensional exception"""
