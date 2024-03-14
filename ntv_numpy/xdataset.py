# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import json
from ntv_numpy.ndarray import NpUtil
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

    def dims(self, var):
        if not var in self.variables: 
            return None
        list_dims = []
        for link in self[var].links:
            list_dims += self.dims(link) if self.dims(link) else [link]
        return list_dims

    @property 
    def validity(self):
        for xn in self:
            if xn.mode in ['relative', 'inconsistent']:
                return 'undefined'
        if self.undef_links or self.undef_vars:
            return 'inconsistent'
        return 'valid'
    
    @property 
    def xtype(self):
        '''Xdataset type'''
        if self.metadata and not (self.additionals or self.variables or 
                                  self.namedarrays):
            return 'meta'        
        match len(self.data_vars):
            case 0:
                return 'group'
            case 1:
                return 'mono'
            case _:
                return 'multi'
            
    @property 
    def dic_xnd(self):
        '''dict of Xndarray'''
        return {xnda.full_name: xnda for xnda in self.xnd}
    
    @property 
    def names(self):
        '''tuple of Xndarray names'''
        return tuple(sorted(xnda.full_name for xnda in self.xnd))
    
    @property 
    def data_arrays(self):
        '''tuple of data_arrays Xndarray names'''
        return (nda for nda in self.namedarrays if not nda in self.dimensions)

    @property 
    def dimensions(self):
        '''tuple of dimensions Xndarray names'''
        dimable = []
        for var in self.variables:
            #dimable += self[var].links
            dimable += self.dims(var)
        return tuple(sorted(set([nda for nda in dimable if nda in self.namedarrays])))
    
    @property 
    def coordinates(self):
        '''tuple of coordinates Xndarray name'''
        dims = set(self.dimensions)
        if not dims:
            return []
        return tuple(sorted(set([xnda.name for xnda in self.xnd 
                if xnda.xtype == 'variable' and set(xnda.links) != dims])))
                #if xnda.xtype == 'variable' and set(self.dims(xnda.name)) != dims])))

    @property 
    def data_vars(self):
        '''tuple of data_vars Xndarray name'''
        dims = set(self.dimensions)
        if not dims:
            return self.variables
        return tuple(sorted(xnda.name for xnda in self.xnd 
                if xnda.xtype == 'variable' and set(xnda.links) == dims))
    
    @property 
    def namedarrays(self):
        '''tuple of namedarray Xndarray name'''
        return tuple(sorted(xnda.name for xnda in self.xnd if xnda.xtype == 'namedarray'))

    @property 
    def variables(self):
        '''tuple of variables Xndarray name'''
        return tuple(sorted(xnda.name for xnda in self.xnd if xnda.xtype == 'variable'))

    @property 
    def undef_vars(self):
        '''tuple of variables Xndarray name with inconsistent shape'''
        return tuple(sorted([var for var in self.variables if self[var].shape != 
                             [len(self[dim]) for dim in self.dims(var)]]))
                             #[len(self[dim]) for dim in self[var].links]]))   
    @property 
    def undef_links(self):
        '''tuple of variables Xndarray name with inconsistent links'''
        return tuple(sorted([link for var in self.variables for link in self[var].links 
                             if not link in self.names]))

    @property 
    def metadata(self):
        '''tuple of metadata name'''
        return tuple(sorted(xnda.name for xnda in self.xnd if xnda.xtype == 'metadata'))

    @property 
    def additionals(self):
        '''tuple of additionals Xndarray name'''
        return tuple(sorted(xnda.full_name for xnda in self.xnd if xnda.xtype == 'additional'))

    def var_group(self, name):
        return tuple(sorted(xnda.full_name for xnda in self.xnd if xnda.name == name))

    def add_group(self, name):
        return tuple(sorted(xnda.full_name for xnda in self.xnd if xnda.add_name == name))
    
    @property 
    def partition(self):
        dic = {}
        dic |= {'data_vars' : list(self.data_vars)} if self.data_vars else {}
        dic |= {'data_arrays' : list(self.data_arrays)} if self.data_arrays else {}
        dic |= {'dimensions' : list(self.dimensions)} if self.dimensions else {}
        dic |= {'coordinates' : list(self.coordinates)} if self.coordinates else {}
        dic |= {'additionals' : list(self.additionals)} if self.additionals else {}
        dic |= {'metadata' : list(self.metadata)} if self.metadata else {}
        return dic    

    @property 
    def info(self):
        inf = {'name': self.name, 'xtype': self.xtype} | self.partition
        inf['validity'] = self.validity
        inf['length'] = len(self[self.data_vars[0]]) if self.data_vars else 0
        inf['width'] = len(self)
        return {key: val for key, val in inf.items() if val}        
        
    @staticmethod
    def read_json(jso, **kwargs):
        ''' convert json data into a Xdataset.
        
        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with 
        non Numpy ntv_type into Xndarray with python type
        '''
        option = {'convert': True} | kwargs
        if not isinstance(jso, dict):
            return None
        if len(jso) == 1:
            json_name, value = list(jso.items())[0]
            name = Xndarray.split_json_name(json_name)[0]
        else:
            value = jso
            name = None
        xnd = [Xndarray.read_json({key: val}, **option) for key, val in value.items()]
        return Xdataset(xnd, name)
            
    def to_json(self, **kwargs):
        ''' convert a Xdataset into json-value.

        *Parameters*

        - **encoded** : Boolean (default False) - json value if False else json text
        - **header** : Boolean (default True) - including 'xdataset' type
        - **notype** : list of Boolean (default list of None) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default False) - if True, without shape if dim < 1
        - **format** : list of string (default list of 'full') - representation format of the ndarray,
        '''            
        notype = kwargs['notype'] if ('notype' in kwargs and isinstance(kwargs['notype'], list) and
                    len(kwargs['notype']) == len(self)) else [False] * len(self)
        format = kwargs['format'] if ('format' in kwargs and isinstance(kwargs['format'], list) and
                    len(kwargs['format']) == len(self)) else ['full'] * len(self)
        noshape = kwargs.get('noshape', False)
        dic_xnd = {}
        for xna, notyp, forma in zip(self.xnd, notype, format):
            dic_xnd |= xna.to_json(notype=notyp, novalue=kwargs.get('novalue', False),
                                   noshape=noshape, format=forma, header=False)
        return NpUtil.json_ntv(self.name, 'xdataset', dic_xnd, 
                               header=kwargs.get('header', True), 
                               encoded=kwargs.get('encoded', False))