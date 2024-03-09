# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import json
from ntv_numpy.ndarray import Ndarray, NpUtil
from ntv_numpy.numpy_ntv_connector import NdarrayConnec
from ntv_numpy.xndarray import Xndarray

class Xdataset:
    ''' Representation of a multidimensional Dataset

    *Attributes :*
    - **name** :  String - name of the Xdataset
    - **xnd**:   list of Xndarray
    '''
    def __init__(self, xnd=None, name=None):    
        '''Xdataset constructor
    
            *Parameters*
    
            - **xnd** : Xdataset/Xndarray/list of Xndarray (default None),
            - **name** : String (default None) - name of the Xdataset
        '''
        self.name = name
        match xnd:
            case list():
                self.xnd = xnd
            case xdat if isinstance(xdat, Xdataset): 
                self.name = xdat.name
                self.xnd  = xdat.xnd
            case xnda if isinstance(xnda, Xndarray): 
                self.xnd = [xnda]
            case _:
                self.xnd = []
        return
        
    def __repr__(self):
        '''return classname and number of value'''
        return self.__class__.__name__ + '[' + str(len(self)) + ']'
    
    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_json())

    def __eq__(self, other):
        '''equal if xnd are equal'''
        for xnda in self.xnd:
            if not xnda in other:
                return False
        for xnda in other.xnd:
            if not xnda in self:
                return False
        return True
     
    def __len__(self):
        '''number of Xndarray'''
        return len(self.xnd)

    def __contains__(self, item):
        ''' item of xnd'''
        return item in self.xnd

    def __getitem__(self, selec):
        ''' return Xndarray or tuple of Xndarray with selec:
            - string : name of a xndarray,
            - integer : index of a xndarray,
            - index selector : index interval
            - tuple : names or index '''
        if selec is None or selec == '' or selec in ([], ()):
            return self
        if isinstance(selec, (list, tuple)) and len(selec) == 1:
            selec = selec[0]
        if isinstance(selec, tuple):
            return [self[i] for i in selec]
        if isinstance(selec, str):
            return self.dic_xnd[selec]
        if isinstance(selec, list):
            return self[selec[0]][selec[1:]]
        return self.xnd[selec]

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)      

    @property 
    def dic_xnd(self):
        '''dict of Xndarray'''
        return {xnda.full_name: xnda for xnda in self.xnd}
    
    @property 
    def names(self):
        '''tuple of Xndarray names'''
        return tuple(xnda.full_name for xnda in self.xnd)
    
    @property 
    def coordinates(self):
        '''tuple of coordinates Xndarray name'''
        dims = set(self.dimensions)
        if not dims:
            return []
        return tuple(set([xnda.name for xnda in self.xnd 
                if xnda.xtype == 'variable' and set(xnda.dims) != dims]))

    @property 
    def data_vars(self):
        '''tuple of data_vars Xndarray name'''
        dims = set(self.dimensions)
        if not dims:
            return self.variables
        return tuple(xnda.name for xnda in self.xnd 
                if xnda.xtype == 'variable' and set(xnda.dims) == dims)
    
    @property 
    def dimensions(self):
        '''tuple of dimensions Xndarray name'''
        return tuple(xnda.name for xnda in self.xnd if xnda.xtype == 'dimension')

    @property 
    def variables(self):
        '''tuple of variables Xndarray name'''
        return tuple(xnda.name for xnda in self.xnd if xnda.xtype == 'variable')

    @property 
    def metadata(self):
        '''tuple of metadata name'''
        return tuple(xnda.name for xnda in self.xnd if xnda.xtype == 'metadata') 

    @property 
    def additionals(self):
        '''tuple of additionals Xndarray name'''
        return tuple(xnda.full_name for xnda in self.xnd if xnda.xtype == 'additional') 

    def var_group(self, name):
        return tuple(xnda.full_name for xnda in self.xnd if xnda.name == name)

    def add_group(self, name):
        return tuple(xnda.full_name for xnda in self.xnd if xnda.add_name == name)
    
    @property 
    def partition(self):
        dic = {}
        dic |= {'data_vars' : list(self.data_vars)} if self.data_vars else {}
        dic |= {'dimensions' : list(self.dimensions)} if self.dimensions else {}
        dic |= {'coordinates' : list(self.coordinates)} if self.coordinates else {}
        dic |= {'metadata' : list(self.metadata)} if self.metadata else {}
        return dic    
    
    @staticmethod
    def read_json(jso, header=True):
        if not isinstance(jso, dict):
            return None
        if header: 
            json_name, value = list(jso.items())[0]
            name = Xndarray.split_json_name(json_name)[0]
        else:
            value = jso
            name = None
        xnd = [Xndarray.read_json({key: val}) for key, val in value.items()]
        return Xdataset(xnd, name)
            
    def to_json(self, **kwargs):
        ''' convert a Xdataset into json-value.

        *Parameters*

        - **encoded** : Boolean (default False) - json value if False else json text
        - **header** : Boolean (default True) - including 'xdataset' type
        - **notype** : list of Boolean (default list of None) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **format** : list of string (default list of 'full') - representation format of the ndarray,
        '''            
        notype = kwargs['notype'] if ('notype' in kwargs and 
                    len(kwargs['notype']) == len(self)) else [False] * len(self)
        format = kwargs['format'] if ('format' in kwargs and 
                    len(kwargs['format']) == len(self)) else ['full'] * len(self)
        dic_xnd = {}
        for xna, notyp, forma in zip(self.xnd, notype, format):
            dic_xnd |= xna.to_json(notype=notyp, novalue=kwargs.get('novalue', False),
                                   format=forma, header=False)
        #return {self.name : dic_xnd}
        return NpUtil.json_ntv(self.name, 'xdataset', dic_xnd, 
                               header=kwargs.get('header', True), 
                               encoded=kwargs.get('encoded', False))