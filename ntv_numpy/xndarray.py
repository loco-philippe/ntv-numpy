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
from ntv_numpy.ndarray import Ndarray, NpUtil, NdarrayError
from json_ntv import Ntv

class Xndarray:
    ''' Representation of a labelled multidimensional Array

    *Attributes :*
    - **name** :  string - name of the Xndarray
    - **add_name** :  string - additional name of the Xndarray
    - **ntv_type**:  string - NTVtype of the Xndarray
    - **nda**: np.ndarray - Numpy ndarray data
    - **uri**: string - address of the nda
    - **links**: list of string - links to other Xndarray
    - **meta** : JsonValue - informations

    *dynamic values (@property)*
    - `shape`
    - `info`
    - `mode`
    - `xtype`
    - `full_name`
    - `json_name`

    *methods*
    - `to_json`
    - `read_json` (staticmethod)
    - `split_json_name` (staticmethod)
    - `split_name` (staticmethod)
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
        #print('init xnd', full_name, nda.to_json2(), links, meta)
        if isinstance(full_name, Xndarray):
            self.name = full_name.name
            self.add_name = full_name.add_name
            self.nda = full_name.nda
            self.links = full_name.links
            self.meta = full_name.meta
            return
        self.name, self.add_name = Xndarray.split_name(full_name)
        self.nda = Ndarray(nda) if not nda is None else None
        self.links = sorted(links) if links else None
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
        if self.name != other.name:
            return False
        if self.links != other.links:
            return False
        if self.meta != other.meta:
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
        inf['xtype'] = self.xtype
        inf['links'] = self.links
        return {key: val for key, val in inf.items() if val}

    @property
    def xtype(self):
        '''nature of the Xndarray (undefined, namedarray, variable, additional,
        inconsistent)'''
        match [self.links, self.add_name, self.mode]:
            case [_, _, 'inconsistent']:
                return 'inconsistent'
            case [_, _, 'undefined']:
                return 'metadata'
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
        value, full_name, ntv_type = Ntv.decode_json(jso)[:3]
        
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
        #print('xnd json', nda, meta, links)
        nda = Ndarray.read_json2(nda, **option) if nda else None
        #print('xnd nda', nda.to_json2())
        return Xndarray(full_name, links=links, meta=meta, nda=nda)

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
        nda_str = Ndarray.to_json2(self.nda, 
                                  **opt_nda) if not self.nda is None else None
        lis = [nda_str, self.links, self.meta]
        lis = [val for val in lis if not val is None]
        return NpUtil.json_ntv(None if option['noname'] else self.full_name,
                               None if option['noname'] else 'xndarray',
                               lis[0] if lis == [self.meta] else lis,
                               header=option['header'], encoded=option['encoded'])

    @staticmethod
    def split_json_name(string, notnone=False):
        '''return a tuple with name, ntv_type from string'''
        null = '' if notnone else None
        if not string or string == ':':
            return (null, null)
        spl = string.rsplit(':', maxsplit=1)
        if len(spl) == 1:
            return (string, null)
        if spl[0] == '':
            return (null, spl[1])
        sp0 = spl[0][:-1] if spl[0][-1] == ':' else spl[0]
        return (null if sp0 == '' else sp0, null if spl[1] == '' else spl[1])

    @staticmethod
    def split_name(string):
        '''return a list with name, add_name from string'''
        if not string or string == '.':
            return ['', '']
        spl = string.split('.', maxsplit=1)
        spl = [spl[0], ''] if len(spl) < 2 else spl
        return spl

    def _to_json(self):
        return {'name': self.name, 'ntv_type': self.ntv_type, 'uri': self.uri,
                'nda': self.nda, 'meta': self.meta, 'links': self.links}

"""
    def __init__(self, darray, shape=None, ntv_type=None):
        '''Ndarray constructor.

        *Parameters*

        - **darray**: Darray or np.ndarray - data to represent
        - **shape** : String or integer (default None) - name or index of another Darray
        - **ntv_type**: string (default None) - NTVtype to apply
        '''
        if isinstance(darray, Ndarray):
            self.darray = darray.darray
            self.uri = darray.uri
            self.shape = darray.shape
            self.ntvtype = darray.ntvtype
            return
        if isinstance(darray, str):
            self.uri = darray
            darray = None
        else:
            self.uri = None
            if isinstance(darray, list): 
                darray = np.array(darray)
            if isinstance(darray, np.ndarray):
                shape_array = darray.shape
            else:
                shape_array = [len(darray)]
        self.shape = shape if shape else shape_array
        self.darray = Darray(darray)
        ntv_type = NpUtil.nda_ntv_type(self.darray) if not (
            ntv_type or self.darray is None) else ntv_type
        self.ntvtype = Datatype(ntv_type)"""
        
"""
    def __init__(self, full_name=None, nda=None, ntv_type=None, links=None,
                 meta=None):
        '''Xndarray constructor.

        *Parameters*

        - **full_name**: string (default None) - name with additional name
        - **nda** : np.ndarray or string (default None) - data (Numpy or URI)
        - **links**: List of string (default None) - dims or other names of associated Xndarray
        - **ntv_type**: string (default None) - ntv_type to apply to data
        - **meta**: dict (default None) - information
        '''
        if isinstance(full_name, Xndarray):
            self.name = full_name.name
            self.add_name = full_name.add_name
            self.ntv_type = full_name.ntv_type
            self.nda = full_name.nda
            self.uri = full_name.uri
            self.links = full_name.links
            self.meta = full_name.meta
            return
        self.name, self.add_name = Xndarray.split_name(full_name)
        if isinstance(nda, str):
            self.nda = None
            self.uri = nda
        else:
            self.nda = nda
            self.uri = None
        ntv_type = NpUtil.nda_ntv_type(self.nda) if not (
            ntv_type or self.nda is None) else ntv_type
        self.ntv_type = ntv_type
        self.links = sorted(links) if links else None
        self.meta = meta if meta else None

    def __repr__(self):
        '''return classname and number of value'''
        return self.__class__.__name__ + '[' + self.full_name + ']'

    def __str__(self):
        '''return json string format'''
        return json.dumps(self.to_json())

    def __eq__(self, other):
        ''' equal if attributes are equal'''
        if self.name != other.name:
            return False
        if self.ntv_type != other.ntv_type:
            return False
        if self.uri != other.uri:
            return False
        if self.links != other.links:
            return False
        if self.meta != other.meta:
            return False
        if self.nda is None and other.nda is None:
            return True
        if self.nda is None or other.nda is None:
            return False
        return Ndarray.equals(self.nda, other.nda)

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
    def shape(self):
        '''return the shape of the ndarray'''
        return list(self.nda.shape) if self.nda is not None else None

    @property
    def info(self):
        ''' infos of the Xndarray'''
        inf = {'name': self.full_name}
        inf['length'] = len(self)
        inf['mode'] = self.mode
        inf['xtype'] = self.xtype
        inf['ntvtype'] = self.ntv_type
        inf['links'] = self.links
        inf['shape'] = self.shape
        return {key: val for key, val in inf.items() if val}

    @property
    def mode(self):
        '''representation mode of the nda/uri data (relative, absolute,
        undefined, inconsistent)'''
        match [self.nda, self.uri]:
            case [None, str()]:
                return 'relative'
            case [None, None]:
                return 'undefined'
            case [_, None]:
                return 'absolute'
            case _:
                return 'inconsistent'

    @property
    def xtype(self):
        '''nature of the Xndarray (undefined, namedarray, variable, additional,
        inconsistent)'''
        match [self.links, self.add_name, self.mode]:
            case [_, _, 'undefined']:
                return 'metadata'
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
        value, full_name, ntv_type = Ntv.decode_json(jso)[:3]
        
        uri = meta = links = str_nda = None
        match value:
            case str(meta) | dict(meta): ...
            case [str(uri)]: ...
            case [str(uri), list(links)]: ...
            case [str(uri), dict(meta)] | [str(uri), str(meta)]: ...
            case [str(uri), list(links), dict(meta)]: ...
            case [str(uri), list(links), str(meta)]: ...
            case [list(str_nda)]: ...
            case [list(str_nda), list(links)]: ...
            case [list(str_nda), dict(meta)] | [list(str_nda), str(meta)]: ...
            case [list(str_nda), list(links), dict(meta)]: ...
            case [list(str_nda), list(links), str(meta)]: ...
            case _:
                return None
        ntv_type = str_nda[0] if str_nda and isinstance(
            str_nda[0], str) else None
        nda = Ndarray.read_json(str_nda, **option) if str_nda else uri
        return Xndarray(full_name, ntv_type=ntv_type, links=links,
                        meta=meta, nda=nda)

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
        option = {'notype': False, 'extension': None, 'format': 'full',
                  'noshape': True, 'header': True, 'encoded': False,
                  'novalue': False, 'noname': False} | kwargs
        if not option['format'] in ['full', 'complete']:
            option['noshape'] = False
        nda_str = Ndarray.to_json(self.nda, ntv_type=self.ntv_type,
                                  **option) if not self.nda is None else None
        lis = [self.uri, nda_str, self.links, self.meta]
        lis = [val for val in lis if not val is None]
        return NpUtil.json_ntv(None if option['noname'] else self.full_name,
                               None if option['noname'] else 'xndarray',
                               lis[0] if lis == [self.meta] else lis,
                               header=option['header'], encoded=option['encoded'])

    @staticmethod
    def split_json_name(string, notnone=False):
        '''return a tuple with name, ntv_type from string'''
        null = '' if notnone else None
        if not string or string == ':':
            return (null, null)
        spl = string.rsplit(':', maxsplit=1)
        if len(spl) == 1:
            return (string, null)
        if spl[0] == '':
            return (null, spl[1])
        sp0 = spl[0][:-1] if spl[0][-1] == ':' else spl[0]
        return (null if sp0 == '' else sp0, null if spl[1] == '' else spl[1])

    @staticmethod
    def split_name(string):
        '''return a list with name, add_name from string'''
        if not string or string == '.':
            return ['', '']
        spl = string.split('.', maxsplit=1)
        spl = [spl[0], ''] if len(spl) < 2 else spl
        return spl

    def _to_json(self):
        return {'name': self.name, 'ntv_type': self.ntv_type, 'uri': self.uri,
                'nda': self.nda, 'meta': self.meta, 'links': self.links}

"""