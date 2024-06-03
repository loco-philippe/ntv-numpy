# -*- coding: utf-8 -*-
"""
The `xdataset` module is part of the `ntv-numpy.ntv_numpy` package.

It contains the classes `Xdataset`, `XdatasetInterface`, `XdatasetCategory` for
the multidimensional dataset.

For more information, see the
[user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html)
 or the [github repository](https://github.com/loco-philippe/ntv-numpy).
"""

from abc import ABC, abstractmethod
import json
import pprint
from json_ntv import Ntv
from ntv_numpy.ndarray import Nutil
from ntv_numpy.xndarray import Xndarray
from ntv_numpy.xconnector import XarrayConnec, ScippConnec, AstropyNDDataConnec
from ntv_numpy.xconnector import PandasConnec


class XdatasetCategory(ABC):
    """category of Xndarray (dynamic tuple of full_name) - see Xdataset docstring"""

    xnd: list = NotImplemented
    names: list = NotImplemented

    @abstractmethod
    def dims(self, var, json_name=False):
        """method defined in Xdataset class"""

    @property
    def data_arrays(self):
        """return a tuple of data_arrays Xndarray full_name"""
        return tuple(
            sorted(
                nda
                for nda in self.namedarrays
                if nda not in self.dimensions + self.uniques
            )
        )

    @property
    def dimensions(self):
        """return a tuple of dimensions Xndarray full_name"""
        dimable = []
        for var in self.variables:
            dimable += self.dims(var)
        return tuple(sorted(set(nda for nda in dimable if nda in self.namedarrays)))

    @property
    def shape(self):
        """return an array with the length of dimensions"""
        return [len(self[dim]) for dim in self.dimensions]

    @property
    def coordinates(self):
        """return a tuple of coordinates Xndarray full_name"""
        dims = set(self.dimensions)
        if not dims:
            return ()
        return tuple(
            sorted(
                set(
                    xnda.name
                    for xnda in self.xnd
                    if xnda.xtype == "variable" and set(xnda.links) != dims
                )
            )
        )

    @property
    def data_vars(self):
        """return a tuple of data_vars Xndarray full_name"""
        dims = set(self.dimensions)
        if not dims:
            return self.variables
        return tuple(
            sorted(
                xnda.name
                for xnda in self.xnd
                if xnda.xtype == "variable" and set(xnda.links) == dims
            )
        )

    @property
    def namedarrays(self):
        """return a tuple of namedarray Xndarray full_name"""
        return tuple(
            sorted(xnda.name for xnda in self.xnd if xnda.xtype == "namedarray")
        )

    @property
    def variables(self):
        """return a tuple of variables Xndarray full_name"""
        return tuple(sorted(xnda.name for xnda in self.xnd if xnda.xtype == "variable"))

    @property
    def undef_vars(self):
        """return a tuple of variables Xndarray full_name with inconsistent shape"""
        return tuple(
            sorted(
                var
                for var in self.variables
                if self[var].shape != [len(self[dim]) for dim in self.dims(var)]
            )
        )

    @property
    def undef_links(self):
        """return a tuple of variables Xndarray full_name with inconsistent links"""
        return tuple(
            sorted(
                link
                for var in self.variables
                for link in self[var].links
                if link not in self.names
            )
        )

    @property
    def masks(self):
        """return a tuple of additional Xndarray full_name with boolean ntv_type"""
        return tuple(
            sorted(
                xnda.full_name
                for xnda in self.xnd
                if xnda.xtype == "additional" and xnda.ntv_type == "boolean"
            )
        )

    @property
    def data_add(self):
        """return a tuple of additional Xndarray full_name with not boolean ntv_type"""
        return tuple(
            sorted(
                xnda.full_name
                for xnda in self.xnd
                if xnda.xtype == "additional" and xnda.ntv_type != "boolean"
            )
        )

    @property
    def metadata(self):
        """return a tuple of metadata Xndarray full_name"""
        return tuple(
            sorted(xnda.full_name for xnda in self.xnd if xnda.xtype == "meta")
        )

    @property
    def uniques(self):
        """return a tuple of unique Xndarray full_name"""
        return tuple(
            full_name for full_name in self.namedarrays if len(self[full_name]) == 1
        )

    @property
    def additionals(self):
        """return a tuple of additionals Xndarray full_name"""
        return tuple(
            sorted(xnda.full_name for xnda in self.xnd if xnda.xtype == "additional")
        )

    def group(self, grp):
        """return a tuple of Xndarray full_name with the same name"""
        if isinstance(grp, str):
            return tuple(
                sorted(
                    xnda.full_name
                    for xnda in self.xnd
                    if grp in (xnda.name, xnda.full_name)
                )
            )
        return tuple(sorted(nam for gr_nam in grp for nam in self.group(gr_nam)))

    def add_group(self, add_name):
        """return a tuple of Xndarray full_name with the same add_name"""
        return tuple(
            sorted(xnda.full_name for xnda in self.xnd if xnda.add_name == add_name)
        )


class XdatasetInterface(ABC):
    """Xdataset interface - see Xdataset docstring"""

    name: str = NotImplemented
    xnd: list = NotImplemented

    @staticmethod
    def read_json(jsn, **kwargs):
        """convert json data into a Xdataset.

        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with
        non Numpy ntv_type into Xndarray with python type
        """
        option = {"convert": True} | kwargs
        jso = json.loads(jsn) if isinstance(jsn, str) else jsn
        value, name = Ntv.decode_json(jso)[:2]

        xnd = [Xndarray.read_json({key: val}, **option)
               for key, val in value.items()]
        return Xdataset(xnd, name)

    def to_json(self, **kwargs):
        """convert a Xdataset into json-value.

        *Parameters*

        - **encoded** : Boolean (default False) - json value if False else json text
        - **header** : Boolean (default True) - including 'xdataset' type
        - **notype** : list of Boolean (default list of None) - including data type if False
        - **noname** : Boolean (default False) - including name if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **format** : list of string (default list of 'full') - representation
        format of the ndarray,
        """
        notype = (kwargs["notype"] if ("notype" in kwargs
                                       and isinstance(kwargs["notype"], list)
                                       and len(kwargs["notype"]) == len(self))
                  else [False] * len(self))
        forma = (kwargs["format"] if ("format" in kwargs
                                      and isinstance(kwargs["format"], list)
                                      and len(kwargs["format"]) == len(self))
                 else ["full"] * len(self))
        noshape = kwargs.get("noshape", True)
        dic_xnd = {}
        for xna, notyp, forma in zip(self.xnd, notype, forma):
            dic_xnd |= xna.to_json(
                notype=notyp,
                novalue=kwargs.get("novalue", False),
                noshape=noshape,
                format=forma,
                header=False,
            )
        return Nutil.json_ntv(
            None if kwargs.get("noname", False) else self.name,
            "xdataset",
            dic_xnd,
            header=kwargs.get("header", True),
            encoded=kwargs.get("encoded", False),
        )
    
    def to_xarray(self, **kwargs):
        """return a xr.DataArray or a xr.Dataset from a Xdataset

        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var,
        return a xr.DataArray
        - **info** : Boolean (default True) - if True, add json representation
        of 'relative' Xndarrays and 'data_arrays' Xndarrays in attrs
        """
        return XarrayConnec.xexport(self, **kwargs)

    @staticmethod
    def from_xarray(xar, **kwargs):
        """return a Xdataset from a DataArray or a Dataset"""
        return XarrayConnec.ximport(xar, Xdataset, **kwargs)

    def to_scipp(self, **kwargs):
        """return a sc.DataArray or a sc.Dataset from a Xdataset

        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var,
        return a DataArray
        - **info** : Boolean (default True) - if True return an additional DataGroup with
        metadata and data_arrays
        - **ntv_type** : Boolean (default True) - if True add ntv_type to the name
        """
        return ScippConnec.xexport(self, **kwargs)

    @staticmethod
    def from_scipp(sci, **kwargs):
        """return a Xdataset from a scipp object DataArray, Dataset or DataGroup"""
        return ScippConnec.ximport(sci, Xdataset, **kwargs)

    def to_nddata(self, **kwargs):
        """return a NDData from a Xdataset"""
        return AstropyNDDataConnec.xexport(self, **kwargs)

    @staticmethod
    def from_nddata(ndd, **kwargs):
        """return a Xdataset from a NDData"""
        return AstropyNDDataConnec.ximport(ndd, Xdataset, **kwargs)

    def to_dataframe(self, **kwargs):
        """return a pd.DataFrame from a Xdataset

        *Parameters*

        - **ntv_type**: Boolean (default True) - if False use full_name else json_name
        - **info**: Boolean (default True) - if True add xdt.info in DataFrame.attrs
        - **dims**: list of string (default None) - order of dimensions full_name to apply
        - **index**: Boolean (default True) - if True, dimensions are translated into indexes
        """
        return PandasConnec.xexport(self, **kwargs)

    @staticmethod
    def from_dataframe(dfr, **kwargs):
        """return a Xdataset from a pd.DataFrame

        *Parameters*

        - dims: list of string (default None) - order of dimensions to apply
        """
        return PandasConnec.ximport(dfr, Xdataset, **kwargs)


class Xdataset(XdatasetCategory, XdatasetInterface):
    """Representation of a multidimensional Dataset

    *Attributes :*
    - **name** :  String - name of the Xdataset
    - **xnd**:   list of Xndarray

    *dynamic values (@property)*
    - `xtype`
    - `validity`
    - `dic_xnd`
    - `partition`
    - `length`
    - `info`

    *methods*
    - `parent`
    - `dims`
    - `shape_dims`
    - `to_canonical`
    - `to_ndarray`
    - `to_darray`

    *XdatasetCategory (@property)*
    - `names`
    - `data_arrays`
    - `dimensions`
    - `coordinates`
    - `data_vars`
    - `namedarrays`
    - `variables`
    - `undef_vars`
    - `undef_links`
    - `masks`
    - `data_add`
    - `meta`
    - `metadata`
    - `uniques`
    - `additionals`
    - `group`
    - `add_group`

    *XdatasetInterface methods *
    - `read_json` (static)
    - `to_json`
    - `from_xarray` (static)
    - `to_xarray`
    - `from_scipp` (static)
    - `to_scipp`
    - `from_nddata` (static)
    - `to_nddata`
    - `from_dataframe` (static)
    - `to_dataframe`
    """

    def __init__(self, xnd=None, name=None):
        """Xdataset constructor

        *Parameters*

        - **xnd** : Xdataset/Xndarray/list of Xndarray (default None),
        - **name** : String (default None) - name of the Xdataset
        """
        self.name = name
        match xnd:
            case list():
                self.xnd = xnd
            case xdat if isinstance(xdat, Xdataset):
                self.name = xdat.name
                self.xnd = xdat.xnd
            case xnda if isinstance(xnda, Xndarray):
                self.xnd = [xnda]
            case _:
                self.xnd = []

    def __repr__(self):
        """return classname and number of value"""
        return (
            self.__class__.__name__
            + "["
            + str(len(self))
            + "]\n"
            + pprint.pformat(self.to_json(novalue=True,
                             header=False, noshape=False))
        )

    def __str__(self):
        """return json string format"""
        return json.dumps(self.to_json())

    def __eq__(self, other):
        """equal if xnd are equal"""
        for xnda in self.xnd:
            if xnda not in other:
                return False
        for xnda in other.xnd:
            if xnda not in self:
                return False
        return True

    def __len__(self):
        """number of Xndarray"""
        return len(self.xnd)

    def __contains__(self, item):
        """item of xnd"""
        return item in self.xnd

    def __getitem__(self, selec):
        """return Xndarray or tuple of Xndarray with selec:
        - string : name of a xndarray,
        - integer : index of a xndarray,
        - index selector : index interval
        - tuple : names or index"""
        if selec is None or selec == "" or selec in ([], ()):
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
        """remove a Xndarray (ind is index, name or tuple of names)."""
        if isinstance(ind, int):
            del self.xnd[ind]
        elif isinstance(ind, str):
            del self.xnd[self.names.index(ind)]
        elif isinstance(ind, tuple):
            ind_n = [self.names[i] if isinstance(i, int) else i for i in ind]
            for i in ind_n:
                del self[i]

    def __copy__(self):
        """Copy all the data"""
        return self.__class__(self)

    def parent(self, var):
        """return the Xndarray parent (where the full_name is equal to the name)"""
        if var.name in self.names:
            return self[var.name]
        return var

    def dims(self, var, json_name=False):
        """return the list of parent namedarrays of the links of a Xndarray

        *parameters*

        - **var**: string - full_name of the Xndarray
        - **json_name**: boolean (defaut False) - if True return json_name else full_name
        """
        if var not in self.names:
            return None
        if self[var].add_name and not self[var].links:
            return self.dims(self[var].name, json_name)
        if var in self.namedarrays:
            return [self[var].json_name if json_name else var]
        if var not in self.variables + self.additionals:
            return None
        list_dims = []
        for link in self[var].links:
            list_dims += (
                self.dims(link, json_name) if self.dims(
                    link, json_name) else [link]
            )
        return list_dims

    def shape_dims(self, var):
        """return a shape with the dimensions associated to the var full_name"""
        return (
            [len(self[dim]) for dim in self.dims(var)]
            if set(self.dims(var)) <= set(self.names)
            else None
        )

    @property
    def validity(self):
        """return the validity state: 'inconsistent', 'undifined' or 'valid'"""
        for xnda in self:
            if xnda.mode in ["relative", "inconsistent"]:
                return "undefined"
        if self.undef_links or self.undef_vars:
            return "inconsistent"
        return "valid"

    @property
    def xtype(self):
        """return the Xdataset type: 'meta', 'group', 'mono', 'multi'"""
        if self.metadata and not (
            self.additionals or self.variables or self.namedarrays
        ):
            return "meta"
        if self.validity != "valid":
            return "group"
        match len(self.data_vars):
            case 0:
                return "group"
            case 1:
                return "mono"
            case _:
                return "multi"

    @property
    def dic_xnd(self):
        """return a dict of Xndarray where key is the full_name"""
        return {xnda.full_name: xnda for xnda in self.xnd}

    @property
    def length(self):
        """return the max length of Xndarray"""
        return max(len(xnda) for xnda in self.xnd)

    @property
    def names(self):
        """return a tuple with the Xndarray full_name"""
        return tuple(xnda.full_name for xnda in self.xnd)

    @property
    def partition(self):
        """return a dict of Xndarray grouped with category"""
        dic = {}
        dic |= {"data_vars": list(self.data_vars)} if self.data_vars else {}
        dic |= {"data_arrays": list(self.data_arrays)
                } if self.data_arrays else {}
        dic |= {"dimensions": list(self.dimensions)} if self.dimensions else {}
        dic |= {"coordinates": list(self.coordinates)
                } if self.coordinates else {}
        dic |= {"additionals": list(self.additionals)
                } if self.additionals else {}
        dic |= {"metadata": list(self.metadata)} if self.metadata else {}
        dic |= {"uniques": list(self.uniques)} if self.uniques else {}
        return dic

    @property
    def info(self):
        """return a dict with Xdataset information"""
        inf = {"name": self.name, "xtype": self.xtype} | self.partition
        inf["validity"] = self.validity
        inf["length"] = len(self[self.data_vars[0]]) if self.data_vars else 0
        inf["width"] = len(self)
        data = {
            name: {key: val for key,
                   val in self[name].info.items() if key != "name"}
            for name in self.names
        }
        return {
            "structure": {key: val for key, val in inf.items() if val},
            "data": {key: val for key, val in data.items() if val},
        }

    @property
    def tab_info(self):
        """return a dict with Xdataset information for tabular interface"""
        info = self.info
        data = info["data"]
        t_info = {}
        if "dimensions" in info["structure"]:
            t_info["dimensions"] = info["structure"]["dimensions"]
        t_info["data"] = {
            name: {
                key: val
                for key, val in data[name].items()
                if key in ["shape", "xtype", "meta", "links"]
            }
            for name in data
        }
        return t_info

    def to_canonical(self):
        """remove optional links of the included Xndarray"""
        for name in self.names:
            if self[name].links in ([self[name].name], [name]):
                self[name].links = None
        for add in self.additionals:
            if self[add].links in [self[self[add].name].links, [self[add].name]]:
                self[add].links = None
        for unic in self.uniques:
            self[unic].links = None
        return self

    def to_ndarray(self, full_name):
        """convert a Xndarray from a Xdataset in a np.ndarray"""
        if self.shape_dims(full_name) is None:
            data = self[full_name].ndarray
        else:
            data = self[full_name].darray.reshape(self.shape_dims(full_name))
        if data.dtype.name[:8] == "datetime":
            data = data.astype("datetime64[ns]")
        return data

    def to_darray(self, full_name):
        """convert a Xndarray from a Xdataset in a flattened np.ndarray"""
        data = self[full_name].darray
        if data.dtype.name[:8] == "datetime":
            data = data.astype("datetime64[ns]")
        return data
