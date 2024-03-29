# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:44:48 2024

@author: a lab in the Air
"""

from ntv_numpy.ndarray import NpUtil, Ndarray
from ntv_numpy.xndarray import Xndarray
import xarray as xr
import scipp as sc
import numpy as np

class XarrayConnec:
    
    @staticmethod
    def xexport(xd, **kwargs):
        '''return a DataArray or a Dataset from a Xdataset
       
        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var, return a DataArray
        '''
        option = {'dataset': True, 'datagroup': True} | kwargs 
        coords = XarrayConnec.to_xr_vars(xd, xd.dimensions + xd.coordinates)
        coords |= XarrayConnec.to_xr_vars(xd, xd.additionals)
        attrs = XarrayConnec.to_xr_attrs(xd, **option)
        if len(xd.data_vars) == 1 and not option['dataset']:
            var_name = xd.data_vars[0]
            #data = xd[var_name].nda
            #data = xd[var_name].nda.darray
            if xd.shape_dims(var_name) is None:
                data = xd[var_name].ndarray
            else:
                data = xd[var_name].darray.reshape(xd.shape_dims(var_name))
            #data = xd[var_name].nda.darray.reshape(xd.shape_dims(var_name))  #!!!

            if data.dtype.name[:8] == 'datetime':
                data = data.astype('datetime64[ns]')
            dims = xd.dims(var_name)
            #attrs |= {'ntv_type': xd[var_name].ntv_type}
            attrs |= {'ntv_type': xd[var_name].nda.ntv_type}
            attrs |= xd[var_name].meta if xd[var_name].meta else {}
            #attrs |= xd[var_name].nda.meta if xd[var_name].nda.meta else {}
            name = var_name if var_name != 'data' else None
            """print('data :', data)
            print('coords:', coords)
            print('dims:', dims)
            print('attrs:', attrs)
            print('name:', name)"""
            return xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs,
                                name=name)
        data_vars = XarrayConnec.to_xr_vars(xd, xd.data_vars)
        xrd = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        #xrd = xrd if option['dataset'] else xrd[list(xrd)[0]]
        return xrd      
    
    @staticmethod 
    def ximport(xar, Xclass, **kwargs):
        '''return a Xdataset from a DataArray or a Dataset'''
        xnd = []
        if xar.attrs:
            attrs = {k: v for k, v in xar.attrs.items() if not k in ['name', 'ntv_type']}
            for name, meta in attrs.items():
                if isinstance(meta, list):
                    xnd += [Xndarray.read_json({name: meta})]
                else:    
                    xnd += [Xndarray(name, meta=meta)]
        for coord in xar.coords:
            xnd += [XarrayConnec.var_xr_to_xnd(xar.coords[coord])]
            if list(xar.coords[coord].dims) == list(xar.dims) and isinstance(xar, xr.Dataset):
                xnd[-1].links = [list(xar.data_vars)[0]]                
        if isinstance(xar, xr.DataArray):
            xnd += [XarrayConnec.var_xr_to_xnd(xar, name='data', add_attrs=False)]
            xd = Xclass(xnd, xar.attrs.get('name'))
            for var in xd.data_vars:
                if var != xar.name and xar.name:
                    xd[var].links = [xar.name]
            return xd.to_canonical()
        for var in xar.data_vars:
            xnd += [XarrayConnec.var_xr_to_xnd(xar.data_vars[var])]
        return Xclass(xnd, xar.attrs.get('name')).to_canonical()
    

    
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
        return Xndarray(full_name, Ndarray(nda, ntv_type), dims, attrs)
        #return Xndarray(full_name, nda, ntv_type, dims, attrs)

    @staticmethod 
    def to_xr_attrs(xd, **option):
        '''return a dict with attributes from a Xdataset'''
        attrs = {meta: xd[meta].meta for meta in xd.metadata}
        attrs |= {'name': xd.name} if xd.name else {}
        if option['datagroup']:
            for name in xd.names: 
                if xd[name].mode == 'relative': 
                    attrs |= xd[name].to_json(header=False)
            for name in xd.data_arrays: 
                attrs |= xd[name].to_json(header=False)
        return attrs        
        
    @staticmethod 
    def to_xr_coord(xd, name):
        '''return a dict with Xarray attributes from a Xndarray defined by his name'''
        #data = xd[name].nda
        #data = xd[name].nda.darray
        data = xd[name].nda.darray.reshape(xd.shape_dims(name))  #!!!
        if data.dtype.name[:8] == 'datetime':
            data = data.astype('datetime64[ns]')
        if name in xd.additionals and not xd[name].links:
            #data = data.reshape(xd[xd[name].name].shape)
            data = data.reshape(xd.shape_dims(xd[name].name))
        dims = tuple(xd.dims(name)) if xd.dims(name) else (xd[name].name)
        meta = {'ntv_type': xd[name].ntv_type} | (xd[name].meta if xd[name].meta else {})
        return {name:(dims, data, meta)}
    
    @staticmethod 
    def to_xr_vars(xd, list_names):
        '''return a dict with Xarray attributes from a list of Xndarray names'''
        arg_vars = {}
        valid_names = [name for name in list_names if xd[name].mode == 'absolute']
        for xnd_name in valid_names:
            arg_vars |= XarrayConnec.to_xr_coord(xd, xnd_name)
        return arg_vars
    
    @staticmethod 
    def xr_add_type(xar):
        if isinstance(xar, xr.DataArray) and not 'ntv_type' in xar.attrs:
            xar.attrs |= {'ntv_type': NpUtil.ntv_type(xar.data.dtype.name)} 
            return
        for coord in xar.coords:
            XarrayConnec.xr_add_type(coord)                
        for var in xar.data_vars:
            XarrayConnec.xr_add_type(var)
        return              
       
class ScippConnec:

    SCTYPE_DTYPE = {'string': 'str'}
    
    def xexport(xd, **kwargs):
        '''return a sc.DataArray or a sc.Dataset from a Xdataset
       
        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var, return a DataArray
        - **datagroup** : Boolean (default True) - if True return a DataGroup with metadata and data_arrays
        - **ntv_type** : Boolean (default True) - if True add ntv-type to the name
        '''
        option = {'dataset': True, 'datagroup':True, 'ntv_type':True} | kwargs 
        coords = dict([ScippConnec.to_scipp_var(xd, name, **option) 
                  for name in xd.coordinates + xd.dimensions
                  if xd[name].mode =='absolute'])
        scd = sc.Dataset(dict([ScippConnec.to_sc_dataarray(xd, name, coords, **option)
                           for name in xd.data_vars 
                           if xd[name].mode =='absolute']))
        scd = scd if option['dataset'] else scd[list(scd)[0]]
        if not option['datagroup']:
            return scd
        sc_name = xd.name if xd.name else 'no_name'
        return sc.DataGroup({sc_name:scd} | ScippConnec.to_scipp_grp(xd, **option))

    @staticmethod 
    def ximport(sc_obj, Xclass, **kwargs):
        '''return a Xdataset from a scipp object DataArray, Dataset or DataGroup'''
        xnd = []
        scd = sc_obj
        xnd_name = None
        if isinstance(sc_obj, sc.DataGroup):
           for obj in sc_obj:
               if isinstance(sc_obj[obj], (sc.Dataset, sc.DataArray)):
                   scd = sc_obj[obj]
                   xnd_name = obj
                   break 
        if isinstance(scd, sc.DataArray):
            scd = sc.Dataset({(scd.name if scd.name else 'no_name'): scd})
        if isinstance(scd, sc.Dataset):        
            for coord in scd.coords:
                xnd += ScippConnec.var_sc_to_xnd(scd.coords[coord], scd, coord)   
            for var in scd:
                for mask in scd[var].masks:
                    m_var = Xndarray.split_json_name(var)[0]
                    xnd += ScippConnec.var_sc_to_xnd(scd[var].masks[mask], scd, mask, m_var)   
                xnd += ScippConnec.var_sc_to_xnd(scd[var].data, scd, var)
        if isinstance(sc_obj, sc.DataGroup):            
            xnd = ScippConnec.grp_sc_to_xnd(sc_obj, xnd)
        return Xclass(xnd, xnd_name).to_canonical()    
        
    @staticmethod 
    def grp_sc_to_xnd(sc_obj, xnd):
        '''return a list of Xndarray from a scipp variable'''
        dic_xnd = {xar.name: xar for xar in xnd}
        for obj in sc_obj:
            name, add_name = Xndarray.split_name(obj)
            match [name, add_name, sc_obj[obj]]:
                case [name, None, list()]:
                    xnd += [Xndarray.read_json({name: sc_obj[obj]})]
                case [name, add_name, sc.Variable()]:
                    xnd += ScippConnec.var_sc_to_xnd(sc_obj[obj], None, add_name, name)
                case [name, _, dict() | str() | list()] if name in dic_xnd:
                    if dic_xnd[name].meta:
                        dic_xnd[name].meta |= sc_obj[obj]
                    else:
                        dic_xnd[name].meta = sc_obj[obj]
                case [name, _, dict() | str() | list()]:
                    xnd += [Xndarray.read_json({name: sc_obj[obj]})]
                case [_, _, _]: ...        
        return xnd
    
    @staticmethod 
    def var_sc_to_xnd(scv, scd=None, sc_name='', var=None):
        '''return a list of Xndarray from a scipp variable
        - var : name
        - sc_name : scipp name'''
        l_xnda = []
        unit = scv.unit.name if scv.unit and not scv.unit in ['dimensionless', 'ns'] else ''
        ext_name, typ1 = Xndarray.split_json_name(sc_name, True)
        var_name, typ2 = Xndarray.split_json_name(var, True)
        full_name = var_name + ('.' if var_name and ext_name else '') + ext_name
        ntv_type_base = typ1 + typ2
        ntv_type = ntv_type_base + ('[' + unit + ']' if unit else '')
        links = [Xndarray.split_json_name(jsn)[0] for jsn in scv.dims]
        if not scd is None and sc_name in scd.coords and scv.dims == scd.dims:
            links = [Xndarray.split_json_name(list(scd)[0])[0]]
        if not scv.variances is None:
            nda = Ndarray(np.array(scv.variances), ntv_type_base)
            #nda = np.array(scv.variances)
            l_xnda.append(Xndarray(full_name + '.variance', nda, links))
            #l_xnda.append(Xndarray(full_name + '.variance', nda, None, links))
        nda = np.array(scv.values, dtype=ScippConnec.SCTYPE_DTYPE.get(str(scv.dtype),                                                                str(scv.dtype)))
        if nda.dtype.name == 'datetime64[ns]' and ntv_type: 
            nda = NpUtil.convert(ntv_type, nda, tojson=False)
        l_xnda.append(Xndarray(full_name, Ndarray(nda, ntv_type), links))
        #l_xnda.append(Xndarray(full_name, nda, ntv_type, links))
        return l_xnda

        
    @staticmethod 
    def to_sc_dataarray(xd, name, coords, **option):
        '''return a scipp.DataArray from a Xdataset.global_var defined by his name'''       
        scipp_name, data = ScippConnec.to_scipp_var(xd, name, **option)
        masks = dict([ScippConnec.to_scipp_var(xd, nam, **option) 
                     for nam in set(xd.var_group(name)) & set(xd.masks)])
        return (scipp_name, sc.DataArray(data, coords=coords, masks=masks))
       
    @staticmethod 
    def to_scipp_grp(xd, **option):
        '''return a dict with metadata, data-array and data_add from a Xdataset'''
        grp = {}
        grp |= dict([ScippConnec.to_scipp_var(xd, name, **option)
                     for name in xd.data_add + xd.data_arrays
                     if xd[name].add_name != 'variance'])
        opt_mask = option | {'grp_mask': True}
        grp |= dict([ScippConnec.to_scipp_var(xd, name, **opt_mask)
                     for name in xd.masks
                     if xd[name].name in xd.names and not xd[name].name in xd.data_vars])
        grp |= { name + '.meta': xd[name].meta for name in xd.names
                if xd[name].meta}
        for name in xd.names: 
            if xd[name].mode == 'relative': 
                grp |= xd[name].to_json(header=False)  
        return grp
        
    @staticmethod 
    def to_scipp_var(xd, name, **kwargs):
        '''return a scipp.Variable from a Xndarray defined by his name'''
        option = {'grp_mask': False, 'ntv_type':True} | kwargs
        #print(name, option['grp_mask'])
        add_name = Xndarray.split_name(name)[1]
        new_n = add_name if name in xd.masks and not option['grp_mask'] else name
        opt_n = option['ntv_type']
        #values = xd[name].nda.reshape(xd.shape_dims(name))
        values = xd[name].nda.darray.reshape(xd.shape_dims(name))
        if values.dtype.name[:8] == 'datetime':
            values = values.astype('datetime64[ns]')
        vari_name = name + '.variance'
        variances = xd[vari_name].nda.darray if vari_name in xd.names else None
        #variances = xd[vari_name].nda if vari_name in xd.names else None
        if not variances is None:
            variances = variances.reshape(xd.shape_dims(vari_name))
        dims = xd.dims(name, opt_n) if xd.dims(name, opt_n) else [xd[name].name]
        simple_type, unit = NpUtil.split_type(xd[name].ntv_type)
        scipp_name = new_n + (':' + simple_type if opt_n else '')
        unit = unit if unit else ''
        #print(scipp_name)
        return (scipp_name, sc.array(dims=dims, values=values, 
                                     variances=variances, unit=unit))
