# -*- coding: utf-8 -*-
"""
@author: Philippe@loco-labs.io

The `ndtype` module is part of the `ntv-numpy.ntv_numpy` package ([specification document](
https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)).

It contains the class `Ndtype` for NTVtype with extension.

For more information, see the
[user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html)
 or the [github repository](https://github.com/loco-philippe/ntv-numpy).
"""
import configparser
from pathlib import Path
import json
from json_ntv import Datatype
import ntv_numpy


class Ndtype(Datatype):
    ''' The Ndtype is a child class of Datatype with additional attributes.

    *Additional attributes :*
    - add_type: additional data added to the JSON ndarray
    - dtype: data type of the np.ndarray

    *Inherited attributes :*
    - **name** : String - name of the Datatype
    - **nspace** : Namespace - namespace associated
    - **custom** : boolean - True if not referenced
    - **validate** : function (default None) - validate function
    - **typebase** : TypeBase - TypeBas of the Datatype
    - **extension** : String (default None) - extension of the TypeBase

    The methods defined in this class are :
    - `read_ini` (static method)
    '''
    @staticmethod
    def read_ini():
        '''return a dict with config data read in ntv_numpy.ini'''
        config = configparser.ConfigParser()
        p_file = Path('ntv_numpy.ini')
        config.read(Path(ntv_numpy.__file__).parent / p_file)
        types = json.loads(config['data']['types'])
        return {ntv_type: {'add_type': add_type, 'dtype': dtype}
                for [ntv_type, add_type, dtype] in types}

    def __init__(self, full_name, module=False, force=False, validate=None):
        '''NdType constructor.

        *Parameters*

        - **full_name** : String - absolut name of the Datatype
        - **module** : boolean (default False) - if True search data in the
        local .ini file, else in the distant repository
        - **force** : boolean (default False) - if True, no Namespace control
        - **validate** : function (default None) - validate function to include'''
        super().__init__(full_name, module=module, force=force, validate=validate)
        np_type = NP_TYPES.get(self.base_name)
        self.dtype = np_type['dtype'] if np_type else None
        self.add_type = np_type['add_type'] if np_type else None


NP_TYPES = Ndtype.read_ini()
NP_NTYPE = {val['dtype']: key for key, val in NP_TYPES.items()}
