# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `xndarray` module is part of the `ntv-numpy.ntv_numpy` package ([specification document](
https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)).

It contains the classes `Xndarray` for the labeled multidimensional array.

For more information, see the
[user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html)
 or the [github repository](https://github.com/loco-philippe/ntv-numpy).
"""

import json
from json_ntv import Ntv
from ntv_numpy.ndarray import Ndarray, Nutil, NdarrayError


class Xndarray:
    ''' Representation of a labelled multidimensional Array

    *Attributes :*
    - **name** :  string - name of the Xndarray
    - **add_name** :  string - additional name of the Xndarray
    - **nda**: Ndarray - ndarray data
    - **links**: list of string - links to other Xndarray
    - **meta** : JsonValue - informations

    *dynamic values (@property)*
    - `darray`
    - `ndarray`
    - `uri`
    - `shape`
    - `ntv_type`
    - `info`
    - `mode`
    - `xtype`
    - `full_name`
    - `json_name`

    *methods*
    - `to_json`
    - `read_json (static method)`
    - `set_ndarray`
    '''

    def __init__(self, full_name, nda=None, links=None,
                 meta=None):
        '''Xndarray constructor.

        *Parameters*

        - **full_name**: string (default None) - name with additional name
        - **nda** : Ndarray (default None) - data
        - **links**: List of string (default None) - dims or other names of associated Xndarray
        - **ntv_type**: string (default None) - ntv_type to apply to data
        - **meta**: dict (default None) - information
        '''
        # print('init xnd', full_name, nda.to_json(), links, meta)
        if isinstance(full_name, Xndarray):
            self.name = full_name.name
            self.add_name = full_name.add_name
            self.nda = full_name.nda
            self.links = full_name.links
            self.meta = full_name.meta
            return
        self.name, self.add_name = Nutil.split_name(full_name)
        self.nda = Ndarray(nda) if not nda is None else None
        self.links = links if links else None
        self.meta = meta if meta else None
        if self.meta is None and self.nda is None:
            raise NdarrayError('A Xndarray has to have metadata or Ndarray')

    def __repr__(self):
        '''return classname and number of value'''
        return self.__class__.__name__ + '[' + self.full_name + ']'

    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_json())

    def __eq__(self, other):
        ''' equal if attributes are equal'''
        if self.name != other.name or self.add_name != other.add_name:
            return False
        if self.links != other.links or self.meta != other.meta:
            return False
        if self.nda is None and other.nda is None:
            return True
        if self.nda is None or other.nda is None:
            return False
        return self.nda == other.nda

    def __len__(self):
        ''' len of ndarray'''
        return len(self.nda) if self.nda is not None else 0

    def __contains__(self, item):
        ''' item of ndarray values'''
        return item in self.nda if self.nda is not None else None

    def __getitem__(self, ind):
        ''' return ndarray value item'''
        if self.nda is None:
            return None
        if isinstance(ind, tuple):
            return [self.nda[i] for i in ind]
            # return [copy(self.values[i]) for i in ind]
        return self.nda[ind]
        # return copy(self.values[ind])

    def __copy__(self):
        ''' Copy all the data '''
        return self.__class__(self)

    @property
    def darray(self):
        '''return the darray of the ndarray'''
        return self.nda.darray if self.nda is not None else None

    @property
    def ndarray(self):
        '''return the darray of the ndarray'''
        return self.nda.ndarray if self.nda is not None else None

    @property
    def uri(self):
        '''return the uri of the ndarray'''
        return self.nda.uri if self.nda is not None else None

    @property
    def shape(self):
        '''return the shape of the ndarray'''
        return self.nda.shape if self.nda is not None else None

    @property
    def ntv_type(self):
        '''return the ntv_type of the ndarray'''
        return self.nda.ntv_type if self.nda is not None else None

    @property
    def mode(self):
        '''return the mode of the ndarray'''
        return self.nda.mode if self.nda is not None else 'undefined'

    @property
    def info(self):
        ''' infos of the Xndarray'''
        inf = {'name': self.full_name}
        inf['length'] = len(self)
        if self.nda:
            inf['mode'] = self.mode
            inf['ntvtype'] = self.ntv_type
            inf['shape'] = self.shape
        inf['uri'] = self.uri
        inf['meta'] = self.meta
        inf['xtype'] = self.xtype
        inf['links'] = self.links
        return {key: val for key, val in inf.items() if val}

    @property
    def xtype(self):
        '''nature of the Xndarray (undefined, namedarray, variable, additional,
        meta, inconsistent)'''
        match [self.links, self.add_name, self.mode]:
            case [_, _, 'inconsistent']:
                return 'inconsistent'
            case [_, _, 'undefined']:
                return 'meta'
            case [None, '', _]:
                return 'namedarray'
            case [_, '', _]:
                return 'variable'
            case [_, str(), _]:
                return 'additional'
            case _:
                return 'inconsistent'

    @property
    def full_name(self):
        '''concatenation of name and additional name'''
        add_name = '.' + self.add_name if self.add_name else ''
        return self.name + add_name

    @property
    def json_name(self):
        '''concatenation of full_name and ntv_type'''
        add_ntv_type = ':' + self.ntv_type if self.ntv_type else ''
        return self.full_name + add_ntv_type

    @staticmethod
    def read_json(jsn, **kwargs):
        ''' convert json data into a Xndarray.

        *Parameters*

        - **convert** : boolean (default True) - If True, convert json data with
        non Numpy ntv_type into data with python type
        '''
        option = {'convert': True} | kwargs
        jso = json.loads(jsn) if isinstance(jsn, str) else jsn
        value, full_name = Ntv.decode_json(jso)[:2]

        meta = links = nda = None
        match value:
            case str(meta) | dict(meta): ...
            case [list(nda)]: ...
            case [list(nda), list(links)]: ...
            case [list(nda), dict(meta)] | [list(nda), str(meta)]: ...
            case [list(nda), list(links), dict(meta)]: ...
            case [list(nda), list(links), str(meta)]: ...
            case _:
                return None
        nda = Ndarray.read_json(nda, **option) if nda else None
        return Xndarray(full_name, links=links, meta=meta, nda=nda)

    def set_ndarray(self, ndarray, nda_uri=True):
        '''set a new ndarray (nda) and return the result (True, False)

        *Parameters*

        - **ndarray** : string, list, np.ndarray, Ndarray - data to include
        - **nda_uri** : boolean (default True) - if True, existing shape and
        ntv_type are not updated (but are created if not existing)'''
        ndarray = Ndarray(ndarray)
        if not self.nda is None:
            return self.nda.update(ndarray, nda_uri=nda_uri)
        self.nda = ndarray
        return True

    def to_json(self, **kwargs):
        ''' convert a Xndarray into json-value.

        *Parameters*

        - **encoded** : Boolean (default False) - json-value if False else json-text
        - **header** : Boolean (default True) - including xndarray type
        - **noname** : Boolean (default False) - including data type and name if False
        - **notype** : Boolean (default False) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **format** : string (default 'full') - representation format of the ndarray,
        - **extension** : string (default None) - type extension
        '''
        option = {'notype': False, 'format': 'full',
                  'noshape': True, 'header': True, 'encoded': False,
                  'novalue': False, 'noname': False} | kwargs
        if not option['format'] in ['full', 'complete']:
            option['noshape'] = False
        opt_nda = option | {'header': False}
        nda_str = self.nda.to_json(**opt_nda) if not self.nda is None else None
        lis = [nda_str, self.links, self.meta]
        lis = [val for val in lis if not val is None]
        return Nutil.json_ntv(None if option['noname'] else self.full_name,
                              None if option['noname'] else 'xndarray',
                              lis[0] if lis == [self.meta] else lis,
                              header=option['header'], encoded=option['encoded'])

    def _to_json(self):
        '''return dict of attributes'''
        return {'name': self.name, 'ntv_type': self.ntv_type, 'uri': self.uri,
                'nda': self.nda, 'meta': self.meta, 'links': self.links}
