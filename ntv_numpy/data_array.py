# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:14:40 2024

@author: a lab in the Air
"""

import numpy as np
from json_ntv import Ntv
from abc import ABC, abstractmethod
#from copy import copy
import json

class Darray(ABC):
    ''' Representation of a one dimensional Array'''
    def __init__(self, data, ref=None, coding=None, dtype=None):
        if isinstance(data, Darray):
            self.data = data.data
            self.ref = data.ref
            self.coding = data.coding
            return
        data = data if isinstance(data, (list, np.ndarray)) else [data]
        if len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
            self.data = np.fromiter(data, dtype='object')
        else:
            self.data = np.array(data, dtype=dtype).reshape(-1)
        self.ref = ref
        self.coding = np.array(coding)
        return
    
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
            #return [copy(self.values[i]) for i in ind]
        return self.values[ind]
        #return copy(self.values[ind])       

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)
    
    @staticmethod 
    def read_json(val, dtype=None):
        val = val if isinstance(val, list) else [val]
        if not val or not isinstance(val[0], list):
            return Dfull(val, dtype=dtype)
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
                return Dcomplete(data, None, coding, dtype=dtype)
            case _:
                return Dfull(val, dtype=dtype)
    
    
    @abstractmethod
    def to_json(self):
        pass
    
    @property
    @abstractmethod
    def values(self):
        pass
    
    @property
    @abstractmethod
    def _len_val(self):
        pass
    
class Dfull(Darray):    
    ''' Representation of a one dimensional Array with full representation'''
    def __init__(self, data, ref=None, coding=None, dtype=None):
        super().__init__(data, None, None, dtype=dtype)
    
    def to_json(self):
        return self.data.tolist()
    
    @property 
    def values(self):
        return self.data
    
    @property
    def _len_val(self):
        return len(self.data)
    

class Dcomplete(Darray):    
    ''' Representation of a one dimensional Array with full representation'''
    def __init__(self, data, ref=None, coding=None, dtype=None):
        if coding is None:
            try:
                data, coding = np.unique(data, return_inverse=True)
            except:
                dat, idx, coding = np.unique(np.frompyfunc(Ntv.from_obj, 1, 1)(data),
                                         return_index=True, return_inverse=True)
                data = data[idx]
        super().__init__(data, None, coding, dtype=dtype)
    
    def to_json(self):
        return [self.data.tolist(), self.coding.tolist()]
    
    @property 
    def values(self):
        return self.data[self.coding]
    
    @property
    def _len_val(self):
        return len(self.coding)
    