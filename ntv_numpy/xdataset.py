# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import json
from ntv_numpy.ndarray import Ndarray
from ntv_numpy.numpy_ntv_connector import NdarrayConnec
from ntv_numpy.xndarray import Xndarray

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
    def coordinates(self):
        dims = set(self.dimensions)
        return [xnda.name for xnda in self.xnd if set(xnda.dims) != dims and xnda.name in self.variables]

    @property 
    def dimensions(self):
        return [xnda.name for xnda in self.xnd if xnda.xtype == 'dimension']

    @property 
    def variables(self):
        return [xnda.name for xnda in self.xnd if xnda.xtype == 'variable']

    @property 
    def metadata(self):
        return [xnda.name for xnda in self.xnd if xnda.xtype == 'metadata']    

    @property 
    def additionals(self):
        return [xnda.full_name for xnda in self.xnd if xnda.xtype == 'additional']    

    @property 
    def var_group(self, name):
        return [xnda.full_name for xnda in self.xnd if xnda.name == name]

    @property 
    def partition(self):
        dic = {}
        dic |= {'variables' : self.variables} if self.variables else {}
        dic |= {'metadata' : self.metadata} if self.metadata else {}
        dic |= {'dimensions' : self.dimensions} if self.dimensions else {}
        return dic    
    
    @staticmethod
    def read_json(jso):
        if not isinstance(jso, dict):
            return None
        json_name, value = list(jso.items())[0]
        name = Xndarray.split_json_name(json_name)[0]
        xnd = [Xndarray.read_json({key: val}) for key, val in value.items()]
        return Xdataset(name, xnd)
            
    def to_json(self, **kwargs):
        ''' convert a Xdataset into json-value.

        *Parameters*

        - **notype** : list of Boolean (default list of None) - including data type if False
        - **format** : list of string (default list of 'full') - representation format of the ndarray,
        '''            
        notype = kwargs['notype'] if ('notype' in kwargs and 
                    len(kwargs['notype']) == len(self)) else [False] * len(self)
        format = kwargs['format'] if ('format' in kwargs and 
                    len(kwargs['format']) == len(self)) else ['full'] * len(self)
        dic_xnd = {}
        for xna, notyp, forma in zip(self.xnd, notype, format):
            dic_xnd |= xna.to_json(notype=notyp, format=forma)
        return {self.name : dic_xnd}
