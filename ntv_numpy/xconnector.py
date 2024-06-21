# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `xconnector` module is part of the `ntv-numpy.ntv_numpy` package ([specification document](
https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)).

It contains interface classes with two static methods `ximport` and `xexport`:
- `XarrayConnec` class for Xarray Dataset or DataArray,
- `AstropyNDDataConnec` class for Astropy NDData,
- `ScippConnec` class for Scipp Dataset or DataArray,
- `PandasConnec` class for pandas dataFrame,


For more information, see the
[user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html)
 or the [github repository](https://github.com/loco-philippe/ntv-numpy).
"""

import importlib

# import xarray as xr
# import scipp as sc
# import pandas as pd
import numpy as np

from ntv_numpy.ndarray import Nutil, Ndarray
from ntv_numpy.xndarray import Xndarray


def import_optional_dependency(name):
    """raise an error if the dependency is not present"""
    msg = (
        f"Missing optional dependency '{name}' " f"Use pip or conda to install {name}."
    )
    try:
        importlib.import_module(name)
    except ImportError as err:
        raise ImportError(msg) from err


class AstropyNDDataConnec:
    """NDData interface with two static methods ximport and xexport"""

    import_optional_dependency("astropy")
    import astropy
    from astropy import wcs
    from astropy.nddata import NDData
    from astropy.nddata.nduncertainty import StdDevUncertainty
    from astropy.nddata.nduncertainty import VarianceUncertainty
    from astropy.nddata.nduncertainty import InverseVariance

    @staticmethod
    def xexport(xdt, **kwargs):
        """return a NDData from a Xdataset"""

        NDData = AstropyNDDataConnec.NDData
        wcs = AstropyNDDataConnec.wcs
        StdDevUncertainty = AstropyNDDataConnec.StdDevUncertainty
        VarianceUncertainty = AstropyNDDataConnec.VarianceUncertainty
        InverseVariance = AstropyNDDataConnec.InverseVariance

        data = xdt["data"].ndarray
        mask = xdt["data.mask"].ndarray
        unit = xdt["data"].nda.ntvtype.extension
        uncert = xdt["data.uncertainty"].ndarray
        typ_u = xdt["data.uncertainty"].nda.ntvtype.extension
        match typ_u:
            case "std":
                uncertainty = StdDevUncertainty(uncert)
            case "var":
                uncertainty = VarianceUncertainty(uncert)
            case "inv":
                uncertainty = InverseVariance(uncert)
            case _:
                uncertainty = uncert
        meta = xdt["meta"].meta | {"name": xdt.name}
        wcs_dic = xdt["wcs"].meta
        psf = xdt["psf"].ndarray
        return NDData(
            data,
            mask=mask,
            unit=unit,
            uncertainty=uncertainty,
            meta=meta,
            wcs=wcs.WCS(wcs_dic),
            psf=psf,
        )

    @staticmethod
    def ximport(ndd, Xclass, **kwargs):
        """return a Xdataset from a astropy.NDData"""
        xnd = []
        name = "no_name"
        unit = ndd.unit.to_string() if ndd.unit is not None else None
        ntv_type = Nutil.ntv_type(ndd.data.dtype.name, ext=unit)
        xnd += [Xndarray("data", nda=Ndarray(ndd.data, ntv_type=ntv_type))]
        if ndd.meta:
            meta = {key: val for key, val in ndd.meta.items() if key != "name"}
            name = ndd.meta.get("name", "no_name")
            xnd += [Xndarray("meta", meta=meta)]
        if ndd.wcs:
            xnd += [Xndarray("wcs", meta=dict(ndd.wcs.to_header()))]
        if ndd.psf is not None:
            xnd += [Xndarray("psf", nda=Ndarray(ndd.psf, ntv_type=ntv_type))]
        if ndd.mask is not None:
            xnd += [Xndarray("data.mask", nda=ndd.mask)]
        if ndd.uncertainty is not None:
            typ_u = ndd.uncertainty.__class__.__name__[:3].lower()
            ntv_type = Nutil.ntv_type(ndd.uncertainty.array.dtype.name, ext=typ_u)
            nda = Ndarray(ndd.uncertainty.array, ntv_type=ntv_type)
            xnd += [Xndarray("data.uncertainty", nda=nda)]
        return Xclass(xnd, name).to_canonical()


class PandasConnec:
    """pandas.DataFrame interface with two static methods ximport and xexport"""

    import_optional_dependency("pandas")
    import pandas

    @staticmethod
    def xexport(xdt, **kwargs):
        """return a pd.DataFrame from a Xdataset

        *Parameters*

        - **ntv_type**: Boolean (default True) - if False use full_name else json_name
        - **info**: Boolean (default True) - if True add xdt.info in DataFrame.attrs
        - **dims**: list of string (default None) - order of dimensions full_name to apply
        - **index**: Boolean (default True) - if True, dimensions are translated into indexes
        """
        pd = PandasConnec.pandas

        opt = {"ntv_type": True, "info": True, "index": True, "dims": None} | kwargs
        dic_name = {
            name: xdt[name].json_name if opt["ntv_type"] else xdt[name].full_name
            for name in xdt.names
        }
        dims = xdt.dimensions if not opt["dims"] else tuple(opt["dims"])
        fields = (
            xdt.group(dims)
            + xdt.group(xdt.coordinates)
            + xdt.group(xdt.data_vars)
            + xdt.uniques
        )
        fields += tuple(
            nam for nam in xdt.group(xdt.data_arrays) if len(xdt[nam]) == xdt.length
        )
        fields_array = tuple(var for var in fields if not xdt[var].uri)
        dic_series = {
            dic_name[name]: PandasConnec._to_np_series(xdt, name, dims)
            for name in fields_array
        }
        dfr = pd.DataFrame(dic_series)
        index = [dic_name[name] for name in dims]
        if index and opt["index"]:
            dfr = dfr.set_index(index)
        if opt["info"]:
            dfr.attrs |= {"info": xdt.tab_info}
            dfr.attrs |= {"metadata": {name: xdt[name].meta for name in xdt.metadata}}
            fields_attrs = [uri for uri in fields if uri not in fields_array] + [
                nam for nam in xdt.group(xdt.data_arrays) if len(xdt[nam]) != xdt.length
            ]
            if fields_attrs:
                dic_fields = {}
                for nam in fields_attrs:
                    dic_fields |= xdt[nam].to_json(header=False)
                dfr.attrs |= {"fields": dic_fields}
            if xdt.name:
                dfr.attrs |= {"name": xdt.name}
        return dfr

    @staticmethod
    def ximport(df, Xclass, **kwargs):
        """return a Xdataset from a pd.DataFrame

        *Parameters*

        - **dims**: list of string (default None) - order of dimensions to apply
        - **distr** : boolean (default True) - True if multi-dimensional analysis
        """
        from time import time
        t0 = time()
        
        opt = {"dims": None, "distr": True} | kwargs
        xnd = []
        dfr = df.reset_index()
        if "index" in dfr.columns and "index" not in df.columns:
            del dfr["index"]
        df_names = {Nutil.split_json_name(j_name)[0]: j_name for j_name in dfr.columns}
        df_ntv_types = {
            Nutil.split_json_name(j_name)[0]: Nutil.split_json_name(j_name)[1]
            for j_name in dfr.columns
        }
        dfr.columns = [Nutil.split_json_name(name)[0] for name in dfr.columns]
        if dfr.attrs.get("metadata"):
            for name, meta in dfr.attrs["metadata"].items():
                xnd += [Xndarray.read_json({name: meta})]
        if dfr.attrs.get("fields"):
            for name, jsn in dfr.attrs["fields"].items():
                xnd += [Xndarray.read_json({name: jsn})]
        if dfr.attrs.get("info"):
            dimensions = dfr.attrs["info"]["dimensions"]
            data = dfr.attrs["info"]["data"]
        else:
            dimensions, data = PandasConnec._ximport_analysis(dfr, opt["dims"],
                                                              distr=opt['distr'])
        
        t1 = time()
        print('t1 : ', t1-t0)
        
        shape_dfr = (
            [data[dim]["shape"][0] for dim in dimensions] if dimensions else len(dfr)
        )

        t2 = time()
        print('t2 : ', t2-t1)

        dfr = dfr.sort_values(dimensions)

        t3 = time()
        print('t3 : ', t3-t2)

        for name in df_names:
            xnd += [
                PandasConnec._ximport_series(
                    data, name, dfr, dimensions, shape_dfr, df_ntv_types, **opt
                )
            ]
        
        t4 = time()
        print('t4 : ', t4-t3)

        dts = Xclass(xnd, dfr.attrs.get("name")).to_canonical()
        
        t5 = time()
        print('t5 : ', t5-t4)
       
        return dts

    @staticmethod
    def _ximport_analysis(dfr, opt_dims, distr=True):
        """return data and dimensions from analysis module
        - **opt_dims**: partition to apply
        - **dfr**: dataframe to analyse
        - **distr** : boolean (default True) - True if multi-dimensional analysis
        """
        from time import time
        t0 = time()
        
        dfr_idx = list(dfr.index.names)
        opt_dims = dfr_idx if dfr_idx != [None] else opt_dims
        ana = dfr.npd.analysis(distr=distr)
        
        t1 = time()
        print('ta1 : ', t1-t0)
        
        partition = ana.field_partition(partition=opt_dims, mode="id")
                        
        t21 = time()
        print('ta21 : ', t21-t1)
 
        part_rel = ana.relation_partition(partition=opt_dims, noroot=True)

                
        t22 = time()
        print('ta22 : ', t22-t21)
        
        part_dim = ana.relation_partition(partition=opt_dims, noroot=True, primary=True)
        
                
        t23 = time()
        print('ta23 : ', t23-t22)
        
        dimensions = partition["primary"]
        len_fields = {fld.idfield: fld.lencodec for fld in ana.fields}
        data = {
            fld.idfield: {
                "shape": [len_fields[dim] for dim in part_dim[fld.idfield]]
                if part_dim
                else [],
                "links": part_rel[fld.idfield] if part_rel else [],
            }
            for fld in ana.fields
        }
        
        t2 = time()
        print('ta2 : ', t2-t1)
        
        for json_name in data:
            if not data[json_name]["shape"]:
                name = Nutil.split_name(Nutil.split_json_name(json_name)[0])[0]
                p_name = [
                    js_name
                    for js_name in data
                    if Nutil.split_json_name(js_name)[0] == name
                ][0]
                data[json_name]["shape"] = data[p_name]["shape"]
        
        t3 = time()
        print('ta3 : ', t3-t2)
        
        return (dimensions, data)

    @staticmethod
    def _ximport_series(data, name, dfr, dimensions, shape_dfr, df_ntv_types, **opt):
        """return a Xndarray from a Series of a pd.DataFrame"""
        if data[name].get("xtype") == "meta":  # or len(dfr[name].unique()) == 1:
            return Xndarray(name, meta=dfr[name].iloc[0])
        meta = data[name].get("meta")
        ntv_type = df_ntv_types[name]
        if len(dfr[name].unique()) == 1:
            nda = Ndarray(np.array(dfr[name].iloc[0]), ntv_type=ntv_type, str_uri=False)
            nda.set_shape([1])
            return Xndarray(name, nda=nda, meta=meta)
        if not dimensions:
            nda = Ndarray(np.array(dfr[name]), ntv_type=ntv_type)
            return Xndarray(name, nda=nda, meta=meta)
        dims = []
        PandasConnec._get_dims(dims, name, data, dimensions)
        if not dims:
            p_name, add_name = Nutil.split_name(name)
            if add_name:
                PandasConnec._get_dims(dims, p_name, data, dimensions)
        np_array = PandasConnec._from_series(
            dfr, name, shape_dfr, dimensions, dims, opt["dims"]
        )
        shape = data[name].get("shape", [len(dfr)])
        nda = Ndarray(np_array, ntv_type, shape)
        links = data[name].get("links")
        return Xndarray(name, nda=nda, links=links if links else dims, meta=meta)

    @staticmethod
    def _to_np_series(xdt, name, dims):
        """return a np.ndarray from the Xndarray of xdt defined by his name

        *parameters*

        - **xdt**: Xdataset - data to convert in a pd.DataFrame
        - **name**: string - full_name of the Xndarray to convert
        - **dims**: list of string - order of dimensions full_name to apply"""
        if name in xdt.uniques:
            return np.array([xdt[name].darray[0]] * xdt.length)
        if xdt[name].shape == [xdt.length]:
            return xdt[name].darray
        n_shape = {nam: len(xdt[nam]) for nam in dims}
        dim_name = xdt.dims(name)
        if not set(dim_name) <= set(dims):
            return None
        add_name = [nam for nam in dims if nam not in dim_name]
        tab_name = add_name + dim_name

        til = 1
        for nam in add_name:
            til *= n_shape[nam]
        shap = [n_shape[nam] for nam in tab_name]
        order = [dims.index(nam) for nam in tab_name]
        arr = xdt[name].darray
        return Nutil.extend_array(arr, til, shap, order)

    @staticmethod
    def _from_series(dfr, name, shape, dims, links, new_dims=None):
        """return a flattened np.ndarray from the pd.Series of dfr defined by his name

        *parameters*

        - dfr: DataFrame - data to convert in Xdataset
        - name: string - name of the Series (full_name or json_name)
        - shape: shape of the Xdataset
        - dims: list of string - list of name of dimensions
        - links: list of string - list of linked Series
        - new_dims: list of string (default None) - new order of dims
        """
        if not links:
            return np.array(dfr[name])
        old_order = list(range(len(dims)))
        new_dims = new_dims if new_dims else dims
        order = [dims.index(dim) for dim in new_dims] if new_dims else old_order
        idx = [0] * len(dims)
        for nam in links:
            idx[new_dims.index(nam)] = slice(shape[dims.index(nam)])
        xar = np.moveaxis(np.array(dfr[name]).reshape(shape), old_order, order)[
            tuple(idx)
        ]
        if not links:
            return xar.flatten()
        lnk = [nam for nam in new_dims if nam in links]
        shape_lnk = [shape[dims.index(nam)] for nam in lnk]
        xar = xar.reshape(shape_lnk)
        old_order = list(range(len(links)))
        order = [lnk.index(dim) for dim in links]
        return np.moveaxis(xar, old_order, order).flatten()

    @staticmethod
    def _get_dims(dims, name, data, dimensions):
        """add names of dimensions into dims"""
        if not name:
            return
        if name in dimensions:
            dims += [name]
        else:
            if "links" not in data[name]:
                return
            for nam in data[name]["links"]:
                PandasConnec._get_dims(dims, nam, data, dimensions)


class XarrayConnec:
    """Xarray interface with two static methods ximport and xexport"""

    import_optional_dependency("xarray")
    import xarray

    @staticmethod
    def xexport(xdt, **kwargs):
        """return a xr.DataArray or a xr.Dataset from a Xdataset

        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var,
        return a xr.DataArray
        - **info** : Boolean (default True) - if True, add json representation
        of 'relative' Xndarrays and 'data_arrays' Xndarrays in attrs
        """
        xr = XarrayConnec.xarray

        option = {"dataset": True, "info": True} | kwargs
        coords = XarrayConnec._to_xr_vars(
            xdt, xdt.dimensions + xdt.coordinates + xdt.uniques
        )
        coords |= XarrayConnec._to_xr_vars(xdt, xdt.additionals)
        attrs = XarrayConnec._to_xr_attrs(xdt, **option)
        if len(xdt.data_vars) == 1 and not option["dataset"]:
            var_name = xdt.data_vars[0]
            data = xdt.to_ndarray(var_name)
            dims = xdt.dims(var_name)
            attrs |= {"ntv_type": xdt[var_name].nda.ntv_type}
            attrs |= xdt[var_name].meta if xdt[var_name].meta else {}
            name = var_name if var_name != "data" else None
            xrd = xr.DataArray(
                data=data, coords=coords, dims=dims, attrs=attrs, name=name
            )
        else:
            data_vars = XarrayConnec._to_xr_vars(xdt, xdt.data_vars)
            xrd = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        for unic in xdt.uniques:
            xrd[unic].attrs |= {"ntv_type": xdt[unic].ntv_type} | (
                xdt[unic].meta if xdt[unic].meta else {}
            )
        return xrd

    @staticmethod
    def ximport(xar, Xclass, **kwargs):
        """return a Xdataset from a xr.DataArray or a xr.Dataset"""
        xr = XarrayConnec.xarray

        xnd = []
        if xar.attrs:
            attrs = {
                k: v for k, v in xar.attrs.items() if k not in ["name", "ntv_type"]
            }
            for name, meta in attrs.items():
                if isinstance(meta, list):
                    xnd += [Xndarray.read_json({name: meta})]
                else:
                    xnd += [Xndarray(name, meta=meta)]
        for coord in xar.coords:
            xnd += [XarrayConnec._var_xr_to_xnd(xar.coords[coord])]
            if list(xar.coords[coord].dims) == list(xar.dims) and isinstance(
                xar, xr.Dataset
            ):
                xnd[-1].links = [list(xar.data_vars)[0]]
        if isinstance(xar, xr.DataArray):
            var = XarrayConnec._var_xr_to_xnd(xar, name="data", add_attrs=False)
            xnd += [XarrayConnec._var_xr_to_xnd(xar, name="data", add_attrs=False)]
            xdt = Xclass(xnd, xar.attrs.get("name"))
            for var in xdt.data_vars:
                if var != xar.name and xar.name:
                    xdt[var].links = [xar.name]
            return xdt.to_canonical()
        for var in xar.data_vars:
            xnd += [XarrayConnec._var_xr_to_xnd(xar.data_vars[var])]
        return Xclass(xnd, xar.attrs.get("name")).to_canonical()

    @staticmethod
    def _var_xr_to_xnd(var, name=None, add_attrs=True):
        """return a Xndarray from a Xarray variable

        *Parameters*

        - **var** : Xarray variable to convert in Xndarray,
        - **name** : string (default None) - default name if var have no name,
        - **add_attrs** : boolean (default True) - if False, attrs are not converted
        """
        full_name = var.name if var.name else name
        name = Nutil.split_name(full_name)[0]
        dims = None if var.dims == (name,) or var.size == 1 else list(var.dims)
        ntv_type = var.attrs.get("ntv_type")
        nda = var.values
        nda = nda.reshape(1) if not nda.shape else nda
        if nda.dtype.name == "datetime64[ns]" and ntv_type:
            nda = Nutil.convert(ntv_type, nda, tojson=False)
        attrs = (
            {k: v for k, v in var.attrs.items() if k not in ["ntv_type", "name"]}
            if add_attrs
            else {}
        )
        return Xndarray(full_name, Ndarray(nda, ntv_type), dims, attrs)

    @staticmethod
    def _to_xr_attrs(xdt, **option):
        """return a dict with attributes from a Xdataset

        *Parameters*

        - **info** : Boolean  if True, add json representation of 'relative'
        Xndarrays and 'data_arrays' Xndarrays in attrs
        """
        attrs = {meta: xdt[meta].meta for meta in xdt.metadata}
        attrs |= {"name": xdt.name} if xdt.name else {}
        if option["info"]:
            for name in xdt.names:
                if xdt[name].mode == "relative":
                    attrs |= xdt[name].to_json(header=False)
            for name in xdt.data_arrays:
                attrs |= xdt[name].to_json(header=False)
        return attrs

    @staticmethod
    def _to_xr_coord(xdt, name):
        """return a dict with Xarray attributes from a Xndarray defined by his name"""
        data = xdt.to_ndarray(name)
        if name in xdt.uniques:
            return {name: data[0]}
        if name in xdt.additionals and not xdt[name].links:
            data = data.reshape(xdt.shape_dims(xdt[name].name))
        dims = tuple(xdt.dims(name)) if xdt.dims(name) else (xdt[name].name)
        meta = {"ntv_type": xdt[name].ntv_type} | (
            xdt[name].meta if xdt[name].meta else {}
        )
        return {name: (dims, data, meta)}

    @staticmethod
    def _to_xr_vars(xdt, list_names):
        """return a dict with Xarray attributes from a list of Xndarray names"""
        arg_vars = {}
        valid_names = [nam for nam in list_names if xdt[nam].mode == "absolute"]
        for xnd_name in valid_names:
            arg_vars |= XarrayConnec._to_xr_coord(xdt, xnd_name)
        for name in list_names:
            if xdt[name].xtype == "meta":
                arg_vars |= {name: xdt[name].meta}
        return arg_vars

    @staticmethod
    def _xr_add_type(xar):
        """add 'ntv_type' as attribute for a xr.DataArray"""
        xr = XarrayConnec.xarray

        if isinstance(xar, xr.DataArray) and "ntv_type" not in xar.attrs:
            xar.attrs |= {"ntv_type": Nutil.ntv_type(xar.data.dtype.name)}
            return
        for coord in xar.coords:
            XarrayConnec._xr_add_type(coord)
        for var in xar.data_vars:
            XarrayConnec._xr_add_type(var)
        return


class ScippConnec:
    """Scipp interface with two static methods ximport and xexport"""

    SCTYPE_DTYPE = {"string": "str"}

    import_optional_dependency("scipp")
    import scipp

    @staticmethod
    def xexport(xdt, **kwargs):
        """return a sc.DataArray or a sc.Dataset from a xdataset

        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var,
        return a DataArray
        - **info** : Boolean (default True) - if True return a DataGroup with
        metadata and data_arrays
        - **ntv_type** : Boolean (default True) - if True add ntv-type to the name
        """
        sc = ScippConnec.scipp

        option = {"dataset": True, "info": True, "ntv_type": True} | kwargs
        coords = dict(
            [
                ScippConnec._to_scipp_var(xdt, name, **option)
                for name in xdt.coordinates + xdt.dimensions + xdt.uniques
                if xdt[name].mode == "absolute"
            ]
        )
        scd = sc.Dataset(
            dict(
                [
                    ScippConnec._to_sc_dataarray(xdt, name, coords, **option)
                    for name in xdt.data_vars
                    if xdt[name].mode == "absolute"
                ]
            )
        )
        scd = scd if option["dataset"] else scd[list(scd)[0]]
        if not option["info"]:
            return scd
        sc_name = xdt.name if xdt.name else "no_name"
        return sc.DataGroup({sc_name: scd} | ScippConnec._to_scipp_grp(xdt, **option))

    @staticmethod
    def ximport(sc_obj, Xclass, **kwargs):
        """return a xdataset from a scipp object DataArray, Dataset or DataGroup"""
        sc = ScippConnec.scipp

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
            scd = sc.Dataset({(scd.name if scd.name else "no_name"): scd})
        if isinstance(scd, sc.Dataset):
            for coord in scd.coords:
                xnd += ScippConnec._var_sc_to_xnd(scd.coords[coord], scd, coord)
            for var in scd:
                for mask in scd[var].masks:
                    m_var = Nutil.split_json_name(var)[0]
                    xnd += ScippConnec._var_sc_to_xnd(
                        scd[var].masks[mask], scd, mask, m_var
                    )
                xnd += ScippConnec._var_sc_to_xnd(scd[var].data, scd, var)
        if isinstance(sc_obj, sc.DataGroup):
            xnd = ScippConnec._grp_sc_to_xnd(sc_obj, xnd)
        return Xclass(xnd, xnd_name).to_canonical()

    @staticmethod
    def _grp_sc_to_xnd(sc_obj, xnd):
        """return a list of Xndarray from a scipp variable"""
        sc = ScippConnec.scipp

        dic_xnd = {xar.name: xar for xar in xnd}
        for obj in sc_obj:
            name, add_name = Nutil.split_name(obj)
            match [name, add_name, sc_obj[obj]]:
                case [name, None, list()]:
                    xnd += [Xndarray.read_json({name: sc_obj[obj]})]
                case [name, add_name, sc.Variable()]:
                    xnd += ScippConnec._var_sc_to_xnd(sc_obj[obj], None, add_name, name)
                case [name, _, dict() | str() | list()] if name in dic_xnd:
                    if dic_xnd[name].meta:
                        dic_xnd[name].meta |= sc_obj[obj]
                    else:
                        dic_xnd[name].meta = sc_obj[obj]
                case [name, _, dict() | str() | list()]:
                    xnd += [Xndarray.read_json({name: sc_obj[obj]})]
                case [_, _, _]:
                    ...
        return xnd

    @staticmethod
    def _var_sc_to_xnd(scv, scd=None, sc_name="", var=None):
        """return a list of Xndarray from a scipp variable
        - scd : scipp dataset
        - scv : scipp variable
        - var : name
        - sc_name : scipp name"""
        l_xnda = []
        unit = (
            scv.unit.name
            if scv.unit and scv.unit not in ["dimensionless", "ns"]
            else ""
        )
        ext_name, typ1 = Nutil.split_json_name(sc_name, True)
        var_name, typ2 = Nutil.split_json_name(var, True)
        full_name = var_name + ("." if var_name and ext_name else "") + ext_name
        ntv_type_base = typ1 + typ2
        ntv_type = ntv_type_base + ("[" + unit + "]" if unit else "")
        links = [Nutil.split_json_name(jsn)[0] for jsn in scv.dims]
        if scd is not None and sc_name in scd.coords and scv.dims == scd.dims:
            links = [Nutil.split_json_name(list(scd)[0])[0]]
        if scv.variances is not None:
            nda = Ndarray(scv.variances, ntv_type_base)
            l_xnda.append(Xndarray(full_name + ".variance", nda, links))
        nda = Ndarray(scv.values, ntv_type, str_uri=False)
        shape = scv.shape if scv.shape else (1,)
        nda.set_shape(shape)
        l_xnda.append(Xndarray(full_name, nda, links))
        return l_xnda

    @staticmethod
    def _to_sc_dataarray(xdt, name, coords, **option):
        """return a scipp.DataArray from a xdataset.global_var defined by his name"""
        sc = ScippConnec.scipp

        scipp_name, data = ScippConnec._to_scipp_var(xdt, name, **option)
        masks = dict(
            [
                ScippConnec._to_scipp_var(xdt, nam, **option)
                for nam in set(xdt.group(name)) & set(xdt.masks)
            ]
        )
        return (scipp_name, sc.DataArray(data, coords=coords, masks=masks))

    @staticmethod
    def _to_scipp_grp(xdt, **option):
        """return a dict with metadata, data-array and data_add from a xdataset"""
        grp = {}
        grp |= dict(
            [
                ScippConnec._to_scipp_var(xdt, name, **option)
                for name in xdt.data_add + xdt.data_arrays
                if xdt[name].add_name != "variance"
            ]
        )
        opt_mask = option | {"grp_mask": True}
        grp |= dict(
            [
                ScippConnec._to_scipp_var(xdt, name, **opt_mask)
                for name in xdt.masks
                if xdt[name].name in xdt.names and xdt[name].name not in xdt.data_vars
            ]
        )
        grp |= {name + ".meta": xdt[name].meta for name in xdt.names if xdt[name].meta}
        for name in xdt.names:
            if xdt[name].mode == "relative":
                grp |= xdt[name].to_json(header=False)
        return grp

    @staticmethod
    def _to_scipp_var(xdt, name, **kwargs):
        """return a scipp.Variable from a Xndarray defined by his name"""
        sc = ScippConnec.scipp

        option = {"grp_mask": False, "ntv_type": True} | kwargs
        simple_type, unit = Nutil.split_type(xdt[name].ntv_type)
        unit = unit if unit else ""
        add_name = Nutil.split_name(name)[1]
        new_n = add_name if name in xdt.masks and not option["grp_mask"] else name
        opt_n = option["ntv_type"]
        scipp_name = new_n + (":" + simple_type if opt_n else "")
        if name in xdt.uniques:
            return (scipp_name, sc.scalar(xdt[name].darray[0], unit=unit))
        vari_name = name + ".variance"
        variances = xdt[vari_name].darray if vari_name in xdt.names else None
        dims = xdt.dims(name, opt_n) if xdt.dims(name, opt_n) else [xdt[name].name]
        var = sc.array(
            dims=["flat"], values=xdt.to_darray(name), variances=variances, unit=unit
        )
        var = sc.fold(var, dim="flat", sizes=dict(zip(dims, xdt[name].shape)))
        return (scipp_name, var)
