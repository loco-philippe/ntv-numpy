# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `xconnector` module is part of the `ntv-numpy.ntv_numpy` package ([specification document](
https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)).

It contains the `XarrayConnec` class for the Xarray interface and the `ScippConnec`
 class for Scipp interface.

For more information, see the
[user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html)
 or the [github repository](https://github.com/loco-philippe/ntv-numpy).
"""

from ntv_numpy.ndarray import NpUtil, Ndarray
from ntv_numpy.xndarray import Xndarray
import xarray as xr
import scipp as sc
import numpy as np


class XarrayConnec:
    ''' Xarray interface with two static methods ximport and xexport'''

    @staticmethod
    def xexport(xdt, **kwargs):
        '''return a xr.DataArray or a xr.Dataset from a Xdataset

        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var,
        return a sc.DataArray
        - **datagroup** : Boolean (default True) - if True, return a sc.DataGroup 
        which contains the sc.DataArray/sc.Dataset and the other data else only
        sc.DataArray/sc.Dataset
        '''
        option = {'dataset': True, 'datagroup': True} | kwargs
        coords = XarrayConnec._to_xr_vars(xdt, xdt.dimensions + xdt.coordinates)
        coords |= XarrayConnec._to_xr_vars(xdt, xdt.additionals)
        attrs = XarrayConnec._to_xr_attrs(xdt, **option)
        if len(xdt.data_vars) == 1 and not option['dataset']:
            var_name = xdt.data_vars[0]
            data = xdt.to_ndarray(var_name)
            dims = xdt.dims(var_name)
            attrs |= {'ntv_type': xdt[var_name].nda.ntv_type}
            attrs |= xdt[var_name].meta if xdt[var_name].meta else {}
            name = var_name if var_name != 'data' else None
            return xr.DataArray(data=data, coords=coords, dims=dims, attrs=attrs,
                                name=name)
        data_vars = XarrayConnec._to_xr_vars(xdt, xdt.data_vars)
        xrd = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        return xrd

    @staticmethod
    def ximport(xar, Xclass, **kwargs):
        '''return a Xdataset from a xr.DataArray or a xr.Dataset'''
        xnd = []    
        if xar.attrs:
            attrs = {k: v for k, v in xar.attrs.items() if not k in [
                'name', 'ntv_type']}
            for name, meta in attrs.items():
                if isinstance(meta, list):
                    xnd += [Xndarray.read_json({name: meta})]
                else:
                    xnd += [Xndarray(name, meta=meta)]
        for coord in xar.coords:
            xnd += [XarrayConnec._var_xr_to_xnd(xar.coords[coord])]
            if list(xar.coords[coord].dims) == list(xar.dims) and isinstance(xar, xr.Dataset):
                xnd[-1].links = [list(xar.data_vars)[0]]
        if isinstance(xar, xr.DataArray):
            var = XarrayConnec._var_xr_to_xnd(xar, name='data', add_attrs=False)
            xnd += [XarrayConnec._var_xr_to_xnd(xar,
                                               name='data', add_attrs=False)]
            xdt = Xclass(xnd, xar.attrs.get('name'))
            for var in xdt.data_vars:
                if var != xar.name and xar.name:
                    xdt[var].links = [xar.name]
            return xdt.to_canonical()
        for var in xar.data_vars:
            xnd += [XarrayConnec._var_xr_to_xnd(xar.data_vars[var])]
        return Xclass(xnd, xar.attrs.get('name')).to_canonical()

    @staticmethod
    def _var_xr_to_xnd(xar, name=None, add_attrs=True):
        '''return a Xndarray from a Xarray variable
        
        *Parameters*

        - **xar** : Xarray variable to convert in Xndarray,
        - **name** : string (default None) - default name if xar have non name,
        - **add_attrs** : boolean (default True) - if False, attrs are not converted
        '''
        full_name = xar.name if xar.name else name
        name = Xndarray.split_name(full_name)[0]
        dims = None if xar.dims == (name,) else list(xar.dims)
        ntv_type = xar.attrs.get('ntv_type')
        nda = xar.values
        if nda.dtype.name == 'datetime64[ns]' and ntv_type:
            nda = NpUtil.convert(ntv_type, nda, tojson=False)
        attrs = {k: v for k, v in xar.attrs.items()
                 if not k in ['ntv_type', 'name']} if add_attrs else {}
        return Xndarray(full_name, Ndarray(nda, ntv_type), dims, attrs)

    @staticmethod
    def _to_xr_attrs(xdt, **option):
        '''return a dict with attributes from a xdataset
        
        *Parameters*

        - **datagroup** : Boolean  if True, add json representation of 'relative'
        Xndarrays and 'data_arrays' Xndarrays 
        '''
        attrs = {meta: xdt[meta].meta for meta in xdt.metadata}
        attrs |= {'name': xdt.name} if xdt.name else {}
        if option['datagroup']:
            for name in xdt.names:
                if xdt[name].mode == 'relative':
                    attrs |= xdt[name].to_json(header=False)
            for name in xdt.data_arrays:
                attrs |= xdt[name].to_json(header=False)
        return attrs

    @staticmethod
    def _to_xr_coord(xdt, name):
        '''return a dict with Xarray attributes from a Xndarray defined by his name'''
        data = xdt.to_ndarray(name)
        if name in xdt.additionals and not xdt[name].links:
            data = data.reshape(xdt.shape_dims(xdt[name].name))
        dims = tuple(xdt.dims(name)) if xdt.dims(name) else (xdt[name].name)
        meta = {'ntv_type': xdt[name].ntv_type} | (
            xdt[name].meta if xdt[name].meta else {})
        return {name: (dims, data, meta)}

    @staticmethod
    def _to_xr_vars(xdt, list_names):
        '''return a dict with Xarray attributes from a list of Xndarray names'''
        arg_vars = {}
        valid_names = [
            name for name in list_names if xdt[name].mode == 'absolute']
        for xnd_name in valid_names:
            arg_vars |= XarrayConnec._to_xr_coord(xdt, xnd_name)
        return arg_vars

    @staticmethod
    def _xr_add_type(xar):
        '''add 'ntv_type' as attribute for a xr.DataArray'''
        if isinstance(xar, xr.DataArray) and not 'ntv_type' in xar.attrs:
            xar.attrs |= {'ntv_type': NpUtil.ntv_type(xar.data.dtype.name)}
            return
        for coord in xar.coords:
            XarrayConnec._xr_add_type(coord)
        for var in xar.data_vars:
            XarrayConnec._xr_add_type(var)
        return


class ScippConnec:
    ''' Scipp interface with two static methods ximport and xexport'''

    SCTYPE_DTYPE = {'string': 'str'}

    @staticmethod
    def xexport(xdt, **kwargs):
        '''return a sc.DataArray or a sc.Dataset from a xdataset

        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var,
        return a DataArray
        - **datagroup** : Boolean (default True) - if True return a DataGroup with
        metadata and data_arrays
        - **ntv_type** : Boolean (default True) - if True add ntv-type to the name
        '''
        option = {'dataset': True, 'datagroup': True,
                  'ntv_type': True} | kwargs
        coords = dict([ScippConnec._to_scipp_var(xdt, name, **option)
                       for name in xdt.coordinates + xdt.dimensions
                       if xdt[name].mode == 'absolute'])
        scd = sc.Dataset(dict([ScippConnec._to_sc_dataarray(xdt, name, coords, **option)
                               for name in xdt.data_vars
                               if xdt[name].mode == 'absolute']))
        scd = scd if option['dataset'] else scd[list(scd)[0]]
        if not option['datagroup']:
            return scd
        sc_name = xdt.name if xdt.name else 'no_name'
        return sc.DataGroup({sc_name: scd} | ScippConnec._to_scipp_grp(xdt, **option))

    @staticmethod
    def ximport(sc_obj, Xclass, **kwargs):
        '''return a xdataset from a scipp object DataArray, Dataset or DataGroup'''
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
                xnd += ScippConnec._var_sc_to_xnd(scd.coords[coord], scd, coord)
            for var in scd:
                for mask in scd[var].masks:
                    m_var = Xndarray.split_json_name(var)[0]
                    xnd += ScippConnec._var_sc_to_xnd(
                        scd[var].masks[mask], scd, mask, m_var)
                xnd += ScippConnec._var_sc_to_xnd(scd[var].data, scd, var)
        if isinstance(sc_obj, sc.DataGroup):
            xnd = ScippConnec._grp_sc_to_xnd(sc_obj, xnd)
        return Xclass(xnd, xnd_name).to_canonical()

    @staticmethod
    def _grp_sc_to_xnd(sc_obj, xnd):
        '''return a list of Xndarray from a scipp variable'''
        dic_xnd = {xar.name: xar for xar in xnd}
        for obj in sc_obj:
            name, add_name = Xndarray.split_name(obj)
            match [name, add_name, sc_obj[obj]]:
                case [name, None, list()]:
                    xnd += [Xndarray.read_json({name: sc_obj[obj]})]
                case [name, add_name, sc.Variable()]:
                    xnd += ScippConnec._var_sc_to_xnd(
                        sc_obj[obj], None, add_name, name)
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
    def _var_sc_to_xnd(scv, scd=None, sc_name='', var=None):
        '''return a list of Xndarray from a scipp variable
        - var : name
        - sc_name : scipp name'''
        l_xnda = []
        unit = scv.unit.name if scv.unit and not scv.unit in [
            'dimensionless', 'ns'] else ''
        ext_name, typ1 = Xndarray.split_json_name(sc_name, True)
        var_name, typ2 = Xndarray.split_json_name(var, True)
        full_name = var_name + \
            ('.' if var_name and ext_name else '') + ext_name
        ntv_type_base = typ1 + typ2
        ntv_type = ntv_type_base + ('[' + unit + ']' if unit else '')
        links = [Xndarray.split_json_name(jsn)[0] for jsn in scv.dims]
        if not scd is None and sc_name in scd.coords and scv.dims == scd.dims:
            links = [Xndarray.split_json_name(list(scd)[0])[0]]
        if not scv.variances is None:
            nda = Ndarray(np.array(scv.variances), ntv_type_base)
            l_xnda.append(Xndarray(full_name + '.variance', nda, links))
        nda = np.array(scv.values, dtype=ScippConnec.SCTYPE_DTYPE.get(str(scv.dtype),
                                                                      str(scv.dtype)))
        if nda.dtype.name == 'datetime64[ns]' and ntv_type:
            nda = NpUtil.convert(ntv_type, nda, tojson=False)
        l_xnda.append(Xndarray(full_name, Ndarray(nda, ntv_type), links))
        return l_xnda

    @staticmethod
    def _to_sc_dataarray(xdt, name, coords, **option):
        '''return a scipp.DataArray from a xdataset.global_var defined by his name'''
        scipp_name, data = ScippConnec._to_scipp_var(xdt, name, **option)
        masks = dict([ScippConnec._to_scipp_var(xdt, nam, **option)
                     for nam in set(xdt.var_group(name)) & set(xdt.masks)])
        return (scipp_name, sc.DataArray(data, coords=coords, masks=masks))

    @staticmethod
    def _to_scipp_grp(xdt, **option):
        '''return a dict with metadata, data-array and data_add from a xdataset'''
        grp = {}
        grp |= dict([ScippConnec._to_scipp_var(xdt, name, **option)
                     for name in xdt.data_add + xdt.data_arrays
                     if xdt[name].add_name != 'variance'])
        opt_mask = option | {'grp_mask': True}
        grp |= dict([ScippConnec._to_scipp_var(xdt, name, **opt_mask)
                     for name in xdt.masks
                     if xdt[name].name in xdt.names and not xdt[name].name in xdt.data_vars])
        grp |= {name + '.meta': xdt[name].meta for name in xdt.names
                if xdt[name].meta}
        for name in xdt.names:
            if xdt[name].mode == 'relative':
                grp |= xdt[name].to_json(header=False)
        return grp

    @staticmethod
    def _to_scipp_var(xdt, name, **kwargs):
        '''return a scipp.Variable from a Xndarray defined by his name'''
        option = {'grp_mask': False, 'ntv_type': True} | kwargs
        add_name = Xndarray.split_name(name)[1]
        new_n = add_name if name in xdt.masks and not option['grp_mask'] else name
        opt_n = option['ntv_type']
        values = xdt.to_ndarray(name)
        vari_name = name + '.variance'
        variances = xdt[vari_name].darray if vari_name in xdt.names else None
        if not variances is None:
            variances = variances.reshape(xdt.shape_dims(vari_name))
        dims = xdt.dims(name, opt_n) if xdt.dims(
            name, opt_n) else [xdt[name].name]
        simple_type, unit = NpUtil.split_type(xdt[name].ntv_type)
        scipp_name = new_n + (':' + simple_type if opt_n else '')
        unit = unit if unit else ''
        return (scipp_name, sc.array(dims=dims, values=values,
                                     variances=variances, unit=unit))
