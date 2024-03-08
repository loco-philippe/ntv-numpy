# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import json
from ntv_numpy.ndarray import Ndarray
from ntv_numpy.numpy_ntv_connector import NdarrayConnec

class Xdataset:
    ''' Representation of a multidimensional labelled Array'''
    def __init__(self, name, xnd=None):    
        if isinstance(name, Xdataset):
            self.name = name.name
            self.xnd = name.xnd
            return
        self.name = name
        self.xnd = xnd if xnd else []
        
    def __repr__(self):
        '''return classname and number of value'''
        return self.__class__.__name__ + '[' + str(len(self)) + ']'
    
    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_json())

    def __eq__(self, other):
        ''' equal if values are equal'''
        for xnda in self.xnd:
            if not xnda in other:
                return False
        for xnda in other.xnd:
            if not xnda in self:
                return False
         return True
     
    def __len__(self):
        ''' len of values'''
        return len(self.xnd)

    def __contains__(self, item):
        ''' item of values'''
        return item in self.xnd

    def __getitem__(self, ind):
        ''' return value item'''
        if isinstance(ind, tuple):
            return [self.xnd[i] for i in ind]
        return self.xnd[ind]

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)      

    @property 
    def dims(self):
        return [xnda.name for xnda in self.xnd if xnda.xtype == 'dimension']
    
    @staticmethod
    def read_json(jso):
        if not isinstance(jso, dict):
            return None
        jso = jso['xdataset'] if 'xdataset' in list(jso)[0] else jso
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
            
    def to_json(self, **kwargs):
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
        return {self.full_name : lis[0] if lis == [self.meta] else lis}

    @property    
    def mode(self):
        match [self.nda, self.uri]:
            case [None, str()]:
                return 'relative'
            case [None, None]:
                return 'undefined'
            case [_, None]:
                return 'absolute'
            case _:
                return 'unconsistent'

    @property    
    def xtype(self):
        match [self.dims, self.add_name, self.mode]:
            case [_, _, 'undefined']:
                return 'metadata'
            case [None, '', _]:
                return 'dimension'      
            case [_, '', _]:
                return 'variable'
            case [_, str(), _]:
                return 'additional'                
            case _:
                return 'unconsistent'
    @property    
    def full_name(self):
        add_name = '.' + self.add_name if self.add_name else ''
        return self.name + add_name
    
            
    def _to_json(self):
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
         
    @staticmethod
    def split_name(string):
        '''return a tuple with name, add_name from string'''
        spl = string.split('.', maxsplit=1)
        spl = [spl[0], ''] if len(spl) < 2 else spl
        return spl