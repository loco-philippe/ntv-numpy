# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import numpy as np
from json_ntv import Ntv
#from abc import ABC, abstractmethod
from copy import copy
import json
from ndarray import Ndarray
from numpy_ntv_connector import NdarrayConnec

class Xndarray:
    ''' Representation of a multidimensional labelled Array'''
    def __init__(self, name, ntv_type=None, nda=None, uri=None, dims=None, meta=None):    
        if isinstance(name, Xndarray):
            self.name = name.name
            self.ntv_type = name.ntv_type
            self.nda = name.nda
            self.uri = name.uri
            self.dims = name.dims
            self.meta = name.meta
            return
        self.name = name
        self.ntv_type = ntv_type
        self.nda = nda
        self.uri = uri
        self.dims = dims
        self.meta = meta      
        
    def __repr__(self):
        '''return classname and number of value'''
        return self.__class__.__name__ + '[' + str(len(self)) + ']'
    
    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_dict())

    def __eq__(self, other):
        ''' equal if values are equal'''
        if self.name != other.name:
            return False
        if self.ntv_type != other.ntv_type:
            return False
        if self.uri != other.uri:
            return False        
        if self.dims != other.dims:
            return False
        if self.meta != other.meta:
            return False
        if self.nda is None and other.nda is None:
            return True
        return Ndarray.equals(self.nda, other.nda)

    def __len__(self):
        ''' len of values'''
        return len(self.nda) if self.nda is not None else 0

    def __contains__(self, item):
        ''' item of values'''
        return item in self.nda

    def __getitem__(self, ind):
        ''' return value item'''
        if isinstance(ind, tuple):
            return [self.nda[i] for i in ind]
            #return [copy(self.values[i]) for i in ind]
        return self.nda[ind]
        #return copy(self.values[ind])       

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)      
    
    @staticmethod
    def read_dict(jso):
        if not (isinstance(jso, dict) and len(jso) == 1):
            return None
        jso = jso['xndarray'] if 'xndarray' in list(jso)[0] else jso
        name = list(jso)[0]
        uri = meta = dims = str_nda = None        
        match jso[name]:
            case str(meta) | dict(meta):...
            case [str(uri)]:...
            case [str(uri), list(dims)]:...
            case [str(uri), dict(meta)] | [str(uri), str(meta)]:...
            case [str(uri), list(dims), dict(meta)]:...
            case [str(uri), list(dims), str(meta)]:...
            case [list(str_nda)]:...
            case [list(str_nda), list(dims)]:...
            case [list(str_nda), dict(meta)] | [list(str_nda), str(meta)]:...
            case [list(str_nda), list(dims), dict(meta)]:...
            case [list(str_nda), list(dims), str(meta)]:...
            case _:
                return None
        ntv_type = str_nda[0] if str_nda and isinstance(str_nda[0], str) else None
        nda = NdarrayConnec.to_obj_ntv(str_nda) if str_nda else None
        return Xndarray(name, ntv_type=ntv_type, uri=uri, dims=dims, meta=meta,
                        nda=nda)
            
    def to_dict(self, **kwargs):
        ''' convert a Xndarray into json-value.

        *Parameters*

        - **notype** : Boolean (default False) - including data type if False
        - **format** : string (default 'full') - representation format of the ndarray,
        - **extension** : string (default None) - type extension
        '''            
        option = {'notype': False, 'extension': None, 'format': 'full', 
                  'noshape': True} | kwargs
        if not option['format'] in ['full', 'complete']: 
            option['noshape'] = False
        nda_str = NdarrayConnec.to_json_ntv(self.nda, typ=self.ntv_type, **option
                                            )[0] if not self.nda is None else None
        lis = [self.uri, nda_str, self.dims, self.meta]
        lis = [val for val in lis if not val is None]
        return {self.name : lis[0] if lis == [self.meta] else lis}
    
    def _to_dict(self):
        return {'name': self.name, 'ntv_type': self.ntv_type, 'uri': self.uri, 'nda': self.nda, 
                'meta': self.meta, 'dims': self.dims}
            
            
    @staticmethod
    def split_json_name(string):
        '''return a tuple with name, ntv_type from string'''
        if not string or string == ':':
            return (None, None)
        spl = string.rsplit(':', maxsplit=1)
        if len(spl) == 1:
            return(string, None)
        if spl[0] == '':
            return (None, spl[1])
        sp0 = spl[0][:-1] if spl[0][-1] == ':' else spl[0]
        return (None if sp0 == '' else sp0, None if spl[1] == '' else spl[1])
         