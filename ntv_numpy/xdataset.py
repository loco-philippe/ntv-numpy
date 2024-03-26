# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:56:11 2024

@author: a lab in the Air
"""

import json
from ntv_numpy.ndarray import NpUtil
from ntv_numpy.xndarray import Xndarray
from ntv_numpy.xconnector import XarrayConnec, ScippConnec


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
                self.xnd = xdat.xnd
            case xnda if isinstance(xnda, Xndarray):
                self.xnd = [xnda]
            case _:
                self.xnd = []

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
            del self.xnd[ind]
        elif isinstance(ind, str):
            del self.xnd[self.names.index(ind)]
        elif isinstance(ind, tuple):
            ind_n = [self.names[i] if isinstance(i, int) else i for i in ind]
            for i in ind_n:
                del self[i]

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)

    def parent(self, var):
        if var.name in self.names:
            return self[var.name]
        return var

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
            list_dims += self.dims(link, json_name) if self.dims(link,
                                                                 json_name) else [link]
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
            dimable += self.dims(var)
        return tuple(sorted(set(nda for nda in dimable if nda in self.namedarrays)))

    @property
    def coordinates(self):
        '''tuple of coordinates Xndarray name'''
        dims = set(self.dimensions)
        if not dims:
            return ()
        return tuple(sorted(set(xnda.name for xnda in self.xnd
                                if xnda.xtype == 'variable' and set(xnda.links) != dims)))

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
        dic |= {'data_vars': list(self.data_vars)} if self.data_vars else {}
        dic |= {'data_arrays': list(self.data_arrays)
                } if self.data_arrays else {}
        dic |= {'dimensions': list(self.dimensions)} if self.dimensions else {}
        dic |= {'coordinates': list(self.coordinates)
                } if self.coordinates else {}
        dic |= {'additionals': list(self.additionals)
                } if self.additionals else {}
        dic |= {'metadata': list(self.metadata)} if self.metadata else {}
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
        xnd = [Xndarray.read_json({key: val}, **option)
               for key, val in value.items()]
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
        - **format** : list of string (default list of 'full') - representation
        format of the ndarray,
        '''
        notype = kwargs['notype'] if ('notype' in kwargs and isinstance(kwargs['notype'], list) and
                                      len(kwargs['notype']) == len(self)) else [False] * len(self)
        forma = kwargs['format'] if ('format' in kwargs and isinstance(kwargs['format'], list) and
                                     len(kwargs['format']) == len(self)) else ['full'] * len(self)
        noshape = kwargs.get('noshape', True)
        dic_xnd = {}
        for xna, notyp, forma in zip(self.xnd, notype, forma):
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
        return XarrayConnec.xexport(self, **kwargs)

    @staticmethod
    def from_xarray(xar, **kwargs):
        '''return a Xdataset from a DataArray or a Dataset'''
        return XarrayConnec.ximport(xar, Xdataset, **kwargs)

    def to_scipp(self, **kwargs):
        '''return a sc.DataArray or a sc.Dataset from a Xdataset

        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var,
        return a DataArray
        - **datagroup** : Boolean (default True) - if True return a DataGroup with
        metadata and data_arrays
        - **ntv_type** : Boolean (default True) - if True add ntv-type to the name
        '''
        return ScippConnec.xexport(self, **kwargs)

    @staticmethod
    def from_scipp(xar, **kwargs):
        '''return a Xdataset from a scipp object DataArray, Dataset or DataGroup'''
        return ScippConnec.ximport(xar, Xdataset, **kwargs)
