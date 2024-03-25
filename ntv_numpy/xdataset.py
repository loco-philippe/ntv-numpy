# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import json
from ntv_numpy.ndarray import NpUtil
from ntv_numpy.xndarray import Xndarray
import xarray as xr
import scipp as sc
import numpy as np

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

    def dims(self, var, json_name=False):
        if not var in self.names: 
            return None
        if self[var].add_name and not self[var].links:
            return self.dims(self[var].name, json_name)
        if var in self.namedarrays:
            return [self[var].json_name if json_name else var]
        if not var in self.variables + self.additionals: 
            return None
        list_dims = []
        for link in self[var].links:
            list_dims += self.dims(link, json_name) if self.dims(link, json_name) else [link]
        return list_dims

    def shape_dims(self, var):
        return [len(self[dim]) for dim in self.dims(var)]
    
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
    def global_vars(self):
        '''tuple of namedarrays or variable Xndarray names'''
        return tuple(sorted(nda for nda in self.namedarrays + self.variables))

    @property 
    def data_arrays(self):
        '''tuple of data_arrays Xndarray names'''
        return tuple(sorted(nda for nda in self.namedarrays if not nda in self.dimensions))

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
    @property 
    def undef_links(self):
        '''tuple of variables Xndarray name with inconsistent links'''
        return tuple(sorted([link for var in self.variables for link in self[var].links 
                             if not link in self.names]))

    @property 
    def masks(self):
        '''tuple of additional Xndarray name with boolean ntv_type'''
        return tuple(sorted([xnda.full_name for xnda in self.xnd 
                             if xnda.xtype == 'additional' and xnda.ntv_type == 'boolean']))

    @property 
    def data_add(self):
        '''tuple of additional Xndarray name with not boolean ntv_type'''
        return tuple(sorted([xnda.full_name for xnda in self.xnd 
                             if xnda.xtype == 'additional' and xnda.ntv_type != 'boolean']))
    
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

    def to_canonical(self):
        '''remove optional dims'''
        for name in self.names:
            if self[name].links in ([self[name].name], [name]):
                self[name].links = None
        for add in self.additionals:
            if self[add].links in [self[self[add].name].links,
                                   [self[add].name]]:
                self[add].links = None
        return self            
    
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
        coords |= Xutil.to_xr_vars(self, self.additionals)
        attrs = {meta: self[meta].meta for meta in self.metadata}
        attrs |= {'name': self.name} if self.name else {}
        if len(self.data_vars) == 1 and not option['dataset']:
            var_name = self.data_vars[0]
            data = self[var_name].nda
            if data.dtype.name[:8] == 'datetime':
                data = data.astype('datetime64[ns]')
            dims = self.dims(var_name)
            attrs |= {'ntv_type': self[var_name].ntv_type}
            attrs |= self[var_name].meta if self[var_name].meta else {}
            name = var_name if var_name != 'data' else None
            return xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs,
                                name=name)
        data_vars = Xutil.to_xr_vars(self, self.data_vars)
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    @staticmethod 
    def from_xarray(xar, **kwargs):
        '''return a Xdataset from a DataArray or a Dataset'''
        xnd = []
        if xar.attrs:
            attrs = {k: v for k, v in xar.attrs.items() if not k in ['name', 'ntv_type']}
            for name, meta in attrs.items():
                xnd += [Xndarray(name, meta=meta)]
        for coord in xar.coords:
            xnd += [Xutil.var_xr_to_xnd(xar.coords[coord])]
            if list(xar.coords[coord].dims) == list(xar.dims) and isinstance(xar, xr.Dataset):
                xnd[-1].links = [list(xar.data_vars)[0]]                
        if isinstance(xar, xr.DataArray):
            xnd += [Xutil.var_xr_to_xnd(xar, name='data', add_attrs=False)]
            xd = Xdataset(xnd, xar.attrs.get('name'))
            for var in xd.data_vars:
                if var != xar.name and xar.name:
                    xd[var].links = [xar.name]
            return xd.to_canonical()
        for var in xar.data_vars:
            xnd += [Xutil.var_xr_to_xnd(xar.data_vars[var])]
        return Xdataset(xnd, xar.attrs.get('name')).to_canonical()
                
    def to_scipp(self, **kwargs):
        '''return a sc.DataArray or a sc.Dataset from a Xdataset
       
        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var, return a DataArray
        - **datagroup** : Boolean (default True) - if True return a DataGroup with metadata and data_arrays
        - **ntv_type** : Boolean (default True) - if True add ntv-type to the name
        '''
        option = {'dataset': True, 'datagroup':True, 'ntv_type':True} | kwargs 
        coords = dict([Xutil.to_scipp_var(self, name, **option) 
                  for name in self.coordinates + self.dimensions
                  if self[name].mode =='absolute'])
        scd = sc.Dataset(dict([Xutil.to_sc_dataarray(self, name, coords, **option)
                           for name in self.data_vars 
                           if self[name].mode =='absolute']))
        scd = scd if option['dataset'] else scd[list(scd)[0]]
        if not option['datagroup']:
            return scd
        return sc.DataGroup({self.name:scd} | Xutil.to_scipp_grp(self, **option))

    @staticmethod 
    def from_scipp(sc_obj, **kwargs):
        '''return a Xdataset from a scipp object DataArray, Dataset or DataGroup'''
        xnd = []
        if isinstance(sc_obj, sc.DataGroup):
           for obj in sc_obj:
               if type(obj) in (sc.Dataset, sc.DataArray):
                   scd = sc_obj[obj]
                   break 
        else:
            scd = sc_obj
        if isinstance(scd, sc.DataArray):
            scd = sc.Dataset({(scd.name if scd.name else 'no_name'): scd})
        for coord in scd.coords:
            xnd += Xutil.var_sc_to_xnd(scd, scd.coords[coord], coord)   
        for var in scd:
            for mask in scd[var].masks:
                m_var = Xndarray.split_json_name(var)[0]
                xnd += Xutil.var_sc_to_xnd(scd, scd[var].masks[mask], mask, m_var)   
            xnd += Xutil.var_sc_to_xnd(scd, scd[var].data, var)
        if isinstance(sc_obj, sc.DataGroup):
           for obj in sc_obj:
               name, add_name = Xndarray.split_name(obj)
               match [name, add_name, type(sc_obj[obj])]:
                   case: 
        return Xdataset(xnd).to_canonical()
                        
class Xutil:
    
    SCTYPE_DTYPE = {'string': 'str'}
    
    @staticmethod 
    def grp_sc_to_xnd(scd, scv, sc_name, var=None):
        '''return a list of Xndarray from a scipp variable'''
        
    @staticmethod 
    def var_sc_to_xnd(scd, scv, sc_name, var=None):
        '''return a list of Xndarray from a scipp variable'''
        l_xnda = []
        unit = scv.unit.name if scv.unit and not scv.unit in ['dimensionless', 'ns'] else ''
        var_n = (var + '.') if var else ''
        json_name = var_n + sc_name + ('[' + unit + ']' if unit else '')
        full_name, ntv_type = Xndarray.split_json_name(json_name)
        links = [Xndarray.split_json_name(jsn)[0] for jsn in scv.dims]
        if sc_name in scd.coords and scv.dims == scd.dims:
            links = [Xndarray.split_json_name(list(scd)[0])[0]]
        if not scv.variances is None:
            nda = np.array(scv.variances)
            l_xnda.append(Xndarray(full_name + '.variance', nda, None, links))
        nda = np.array(scv.values, dtype=Xutil.SCTYPE_DTYPE.get(str(scv.dtype),                                                                str(scv.dtype)))
        if nda.dtype.name == 'datetime64[ns]' and ntv_type: 
            nda = NpUtil.convert(ntv_type, nda, tojson=False)
        l_xnda.append(Xndarray(full_name, nda, ntv_type, links))
        return l_xnda

        
    @staticmethod 
    def to_sc_dataarray(xd, name, coords, **option):
        '''return a scipp.DataArray from a Xdataset.global_var defined by his name'''       
        scipp_name, data = Xutil.to_scipp_var(xd, name, **option)
        masks = dict([Xutil.to_scipp_var(xd, name, **option) 
                     for name in set(xd.var_group(name)) & set(xd.masks)])
        return (scipp_name, sc.DataArray(data, coords=coords, masks=masks))
       
    @staticmethod 
    def to_scipp_grp(xd, **option):
        '''return a dict with metadata, data-array and data_add from a Xdataset'''
        grp = {}
        grp |= dict([Xutil.to_scipp_var(xd, name, **option)
                     for name in xd.data_add + xd.data_arrays
                     if xd[name].add_name != 'variance'])
        grp |= { name + '.meta': xd[name].meta for name in xd.names
                if xd[name].meta}
        for name in xd.names: 
            if xd[name].mode == 'relative': 
                grp |= xd[name].to_json()  
        return grp
        
    @staticmethod 
    def to_scipp_var(xd, name, **option):
        '''return a scipp.Variable from a Xndarray defined by his name'''
        add_name = Xndarray.split_name(name)[1]
        new_n = add_name if name in xd.masks else name
        opt_n = option['ntv_type']
        values = xd[name].nda.reshape(xd.shape_dims(name))
        if values.dtype.name[:8] == 'datetime':
            values = values.astype('datetime64[ns]')
        vari_name = name + '.variance'
        variances = xd[vari_name].nda if vari_name in xd.names else None
        if not variances is None:
            variances = variances.reshape(xd.shape_dims(vari_name))
        dims = xd.dims(name, opt_n) if xd.dims(name, opt_n) else [xd[name].name]
        simple_type, unit = NpUtil.split_type(xd[name].ntv_type)
        scipp_name = new_n + (':' + simple_type if opt_n else '')
        unit = unit if unit else ''
        return (scipp_name, sc.array(dims=dims, values=values, 
                                     variances=variances, unit=unit))

    
    @staticmethod 
    def var_xr_to_xnd(xar, name=None, add_attrs=True):
        '''return a Xndarray from a Xarray variable'''
        full_name = xar.name if xar.name else name
        name, add_name = Xndarray.split_name(full_name)
        dims = None if xar.dims == (name,) else list(xar.dims)
        ntv_type = xar.attrs.get('ntv_type')
        nda = xar.values
        if nda.dtype.name == 'datetime64[ns]' and ntv_type: 
            nda = NpUtil.convert(ntv_type, nda, tojson=False)
        attrs = {k: v for k, v in xar.attrs.items() 
                 if not k in ['ntv_type', 'name']} if add_attrs else {}
        return Xndarray(full_name, nda, ntv_type, dims, attrs)

    @staticmethod 
    def to_xr_coord(xd, name):
        '''return a dict with Xarray attributes from a Xndarray defined by his name'''
        data = xd[name].nda
        if data.dtype.name[:8] == 'datetime':
            data = data.astype('datetime64[ns]')
        if name in xd.additionals and not xd[name].links:
            data = data.reshape(xd[xd[name].name].shape)
        dims = tuple(xd.dims(name)) if xd.dims(name) else (xd[name].name)
        meta = {'ntv_type': xd[name].ntv_type} | (xd[name].meta if xd[name].meta else {})
        return {name:(dims, data, meta)}
    
    @staticmethod 
    def to_xr_vars(xd, list_names):
        '''return a dict with Xarray attributes from a list of Xndarray names'''
        arg_vars = {}
        for xnd_name in list_names:
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
       
