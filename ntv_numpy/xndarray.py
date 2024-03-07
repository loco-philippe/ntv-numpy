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

class Xndarray:
    ''' Representation of a multidimensional labelled Array'''
    def __init__(self, name, nda=None, uri=None, dims=None, meta=None):    
        if isinstance(name, Xndarray):
            self.name = name.name
            self.nda = name.nda
            self.uri = name.uri
            self.dims = name.dims
            self.meta = name.meta
            return
        self.name = name
        self.nda = nda
        self.uri = uri
        self.dims = dims
        self.meta = meta      
        
    def __repr__(self):
        '''return classname and number of value'''
        return self.__class__.__name__ + '[' + str(len(self)) + ']'
    
    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_list())

    def __eq__(self, other):
        ''' equal if values are equal'''
        if self.name != other.name:
            return False
        if self.uri != other.uri:
            return False        
        if self.dims != other.dims:
            return False
        if self.meta != other.meta:
            return False
        return Ndarray.equals(self.nda, other.nda)

    def __len__(self):
        ''' len of values'''
        return len(self.nda) if self.nda else 0

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
        match jso[name]:
            case str(meta):
                return Xndarray(name, meta=meta)
            case dict(dic):
                return Xndarray(name, meta=dic)
            
            
            
            
            