# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import json
from ntv_numpy.ndarray import NpUtil
from ntv_numpy.xndarray import Xndarray
import xarray as xr

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

    def __delitem__(self, ind):
        '''remove a Xndarray (ind is index, name or tuple of names).'''
        if isinstance(ind, int):
            del(self.xnd[ind])
        elif isinstance(ind, str):
            del(self.xnd[self.names.index(ind)])
        elif isinstance(ind, tuple):
            ind_n = [self.names[i] if isinstance(i, int) else i for i in ind]
            for i in ind_n:
                del self[i]
        
    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)      

    def dims(self, var):
        if not var in self.names: 
            return None
        if self[var].add_name:
            return self.dims(self[var].name)
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
        if self.validity != 'valid':
            return 'group'
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
        return tuple(xnda.full_name for xnda in self.xnd)
    
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
            return ()
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
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **format** : list of string (default list of 'full') - representation format of the ndarray,
        '''            
        notype = kwargs['notype'] if ('notype' in kwargs and isinstance(kwargs['notype'], list) and
                    len(kwargs['notype']) == len(self)) else [False] * len(self)
        format = kwargs['format'] if ('format' in kwargs and isinstance(kwargs['format'], list) and
                    len(kwargs['format']) == len(self)) else ['full'] * len(self)
        noshape = kwargs.get('noshape', True)
        dic_xnd = {}
        for xna, notyp, forma in zip(self.xnd, notype, format):
            dic_xnd |= xna.to_json(notype=notyp, novalue=kwargs.get('novalue', False),
                                   noshape=noshape, format=forma, header=False)
        return NpUtil.json_ntv(self.name, 'xdataset', dic_xnd, 
                               header=kwargs.get('header', True), 
                               encoded=kwargs.get('encoded', False))

    def to_xarray(self, **kwargs):
        '''return a DataArray or a Dataset from a Xdataset
       
        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var, return a DataArray
        '''
        option = {'dataset': True} | kwargs 
        coords = Xutil.to_xr_vars(self, self.dimensions + self.coordinates)
        attrs = {meta: self[meta].meta for meta in self.metadata}
        attrs |= {'name': self.name} if self.name else {}
        if len(self.data_vars) == 1 and not option['dataset']:
            var_name = self.data_vars[0]
            data = self[var_name].nda
            if data.dtype.name[:8] == 'datetime':
                data = data.astype('datetime64[ns]')
            dims = self.dims(var_name)
            attrs |= {'ntv_type': self[var_name].ntv_type}
            #if NpUtil.ntv_type(data.dtype.name) != self[var_name].ntv_type:            
            #    attrs |= {'ntv_type': self[var_name].ntv_type}
            attrs |= self[var_name].meta if self[var_name].meta else {}
            name = var_name if var_name != 'no_name' else None
            return xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs,
                                name=name)
        data_vars = Xutil.to_xr_vars(self, self.data_vars)
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    @staticmethod 
    def from_xarray(xar, **kwargs):
        '''return a Xdataset for a DataArray or a Dataset
       
        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var, return a DataArray
        '''
        xnd = []
        if isinstance(xar, xr.DataArray):
            xnd += [Xutil.to_xndarray(xar, name='no_name')]
            for coord in xar.coords:
                xnd += [Xutil.to_xndarray(xar.coords[coord])]
            xd = Xdataset(xnd, xar.attrs.get('name'))
            for var in xd.data_vars:
                if var != xar.name and xar.name:
                    xd[var].links = [xar.name]
            return xd
        for coord in xar.coords:
            xnd += [Xutil.to_xndarray(xar.coords[coord])]
            if list(xar.coords[coord].dims) == list(xar.dims):
                xnd[-1].links = [list(xar.data_vars)[0]]                
        for var in xar.data_vars:
            xnd += [Xutil.to_xndarray(xar.data_vars[var])]
        if xar.attrs:
            attrs = {k: v for k, v in xar.attrs.items() if not k == 'name'}
            for name, meta in attrs.items():
                xnd += [Xndarray(name, meta=meta)]
        return Xdataset(xnd, xar.attrs.get('name'))
                
        
class Xutil:

    @staticmethod 
    def to_xndarray(xar, name=None):
        '''return a Xndarray from a Xarray variable'''
        full_name = xar.name if xar.name else name
        name, add_name = Xndarray.split_name(full_name)
        dims = None if add_name or xar.dims == (name,) else list(xar.dims)
        ntv_type = xar.attrs.get('ntv_type')
        nda = xar.values
        if nda.dtype.name == 'datetime64[ns]' and ntv_type: 
            nda = NpUtil.convert(ntv_type, nda, tojson=False)
        attrs = {k: v for k, v in xar.attrs.items() if not k in ['ntv_type', 'name']}
        #return Xndarray(full_name, nda, ntv_type, dims, None, attrs)
        return Xndarray(full_name, nda, ntv_type, dims, attrs)

    @staticmethod 
    def to_xr_coord(xd, name):
        '''return a dict with Xarray attributes from a Xndarray defined by his name'''
        data = xd[name].nda
        if data.dtype.name[:8] == 'datetime':
            data = data.astype('datetime64[ns]')
        dims = tuple(xd.dims(name)) if xd.dims(name) else (xd[name].name)
        meta = {'ntv_type': xd[name].ntv_type} | (xd[name].meta if xd[name].meta else {})
        #if NpUtil.ntv_type(data.dtype.name) != xd[name].ntv_type:            
        #    meta |= {'ntv_type': xd[name].ntv_type}
        return {name:(dims, data, meta)}
    
    @staticmethod 
    def to_xr_vars(xd, list_names):
        '''return a dict with Xarray attributes from a list of Xndarray names'''
        arg_vars = {}
        grps = [xd.var_group(name) for name in list_names]
        vars_names = [name for grp in grps for name in grp]
        for xnd_name in vars_names:
            arg_vars |= Xutil.to_xr_coord(xd, xnd_name)
        return arg_vars
    
    @staticmethod 
    def xr_add_type(xar):
        if isinstance(xar, xr.DataArray) and not 'ntv_type' in xar.attrs:
            xar.attrs |= {'ntv_type': NpUtil.ntv_type(xar.data.dtype.name)} 
            return
        for coord in xar.coords:
            Xutil.xr_add_type(coord)                
        for var in xar.data_vars:
            Xutil.xr_add_type(var)
        return              
       
