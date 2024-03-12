# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import json
from ntv_numpy.ndarray import Ndarray, NpUtil
#from ntv_numpy.numpy_ntv_connector import NdarrayConnec

class Xndarray:
    ''' Representation of a multidimensional labelled Array'''
    def __init__(self, full_name=None, ntv_type=None, nda=None, uri=None, dims=None, meta=None):    
        if isinstance(full_name, Xndarray):
            self.name = full_name.name
            self.add_name = full_name.add_name
            self.ntv_type = full_name.ntv_type
            self.nda = full_name.nda
            self.uri = full_name.uri
            self.dims = full_name.dims
            self.meta = full_name.meta
            return
        self.name, self.add_name = Xndarray.split_name(full_name)
        ntv_type = NpUtil.nda_ntv_type(nda) if not (ntv_type or nda is None) else ntv_type
        self.ntv_type = ntv_type
        self.nda = nda
        self.uri = uri
        self.dims = dims
        self.meta = meta      
        
    def __repr__(self):
        '''return classname and number of value'''
        #return self.__class__.__name__ + '[' + str(len(self)) + ']'
        return self.__class__.__name__ + '[' + self.full_name + ']'
    
    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_json())

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
        return item in self.nda if self.nda is not None else None

    def __getitem__(self, ind):
        ''' return value item'''
        if self.nda is None:
            return None
        if isinstance(ind, tuple):
            return [self.nda[i] for i in ind]
            #return [copy(self.values[i]) for i in ind]
        return self.nda[ind]
        #return copy(self.values[ind])       

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)      

    @property 
    def shape(self):
        return self.nda.shape if self.nda is not None else None
    
    @staticmethod
    def read_json(jso, **kwargs):
        ''' convert json data into a Xndarray.
        
        
        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with 
        non Numpy ntv_type into data with python type
        '''
        option = {'convert': True} | kwargs
        if not ((isinstance(jso, dict) and len(jso) == 1) or isinstance(jso, list)):
            return None
        if isinstance(jso, list):
            json_name = None
            value = jso
        else:
            json_name, value = list(jso.items())[0]
        full_name = Xndarray.split_json_name(json_name)[0]
        uri = meta = dims = str_nda = None        
        match value:
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
        nda = Ndarray.read_json(str_nda, **option) if str_nda else None
        #nda = NdarrayConnec.to_obj_ntv(str_nda, ) if str_nda else None
        return Xndarray(full_name, ntv_type=ntv_type, uri=uri, dims=dims, 
                        meta=meta, nda=nda)
            
    def to_json(self, **kwargs):
        ''' convert a Xndarray into json-value.

        *Parameters*

        - **encoded** : Boolean (default False) - json-value if False else json-text
        - **header** : Boolean (default True) - including xndarray type
        - **noname** : Boolean (default False) - including data type and name if False
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **format** : string (default 'full') - representation format of the ndarray,
        - **extension** : string (default None) - type extension
        '''            
        option = {'notype': False, 'extension': None, 'format': 'full', 
                  'noshape': True, 'header': True, 'encoded': False,
                  'novalue': False, 'noname': False} | kwargs
        if not option['format'] in ['full', 'complete']: 
            option['noshape'] = False
        nda_str = Ndarray.to_json(self.nda, ntv_type=self.ntv_type, 
                                  **option) if not self.nda is None else None
        #nda_str = NdarrayConnec.to_json_ntv(self.nda, typ=self.ntv_type, **option
        #                                    )[0] if not self.nda is None else None
        lis = [self.uri, nda_str, self.dims, self.meta]
        lis = [val for val in lis if not val is None]
        #jsn = {self.full_name : lis[0] if lis == [self.meta] else lis}
        #name = None if option['noname'] else self.full_name
        #typ = None if option['noname'] else 'xndarray'
        return NpUtil.json_ntv(None if option['noname'] else self.full_name,
                               None if option['noname'] else 'xndarray', 
                               lis[0] if lis == [self.meta] else lis, 
                               header=option['header'], encoded=option['encoded'])
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
        '''return a list with name, add_name from string'''
        if not string or string == '.':
            return ['','']
        spl = string.split('.', maxsplit=1)
        spl = [spl[0], ''] if len(spl) < 2 else spl
        return spl