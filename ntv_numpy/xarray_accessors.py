# -*- coding: utf-8 -*-
"""
The `xarray_accessors` module is part of the `ntv-numpy.ntv_numpy` package.

It contains the class `NnpDatasetAccessors`.

For more information, see the
[user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html)
 or the [github repository](https://github.com/loco-philippe/ntv-numpy).
"""

import xarray as xr

from ntv_numpy.xdataset import Xdataset

try:
    # delete the accessor to avoid warning
    del xr.Dataset.nxr
except AttributeError:
    pass


@xr.register_dataset_accessor("nxr")
class NxrDatasetAccessor:
    """Accessor class for methods invoked as `xr.Dataset.nxr.*`"""

    def __init__(self, xarray_obj):
        """initialisation of the class"""
        self._obj = xarray_obj

    def to_dataframe(self, **kwargs):
        """Accessor for method `Xdataset.from_xarray.to_dataframe` invoked as
        xr.Dataset.nxr.to_dataframe`.

        *Parameters*

        - **ntv_type**: Boolean (default True) - if False use full_name else json_name
        - **info**: Boolean (default True) - if True add xdt.info in DataFrame.attrs
        - **dims**: list of string (default None) - order of dimensions full_name to apply
        - **index**: Boolean (default True) - if True, dimensions are translated into indexes
        """
        return Xdataset.from_xarray(self._obj, **kwargs).to_dataframe(**kwargs)

    def to_scipp(self, **kwargs):
        """Accessor for method `Xdataset.from_xarray.to_scipp` invoked as
        xr.Dataset.nxr.to_scipp`.

        *Parameters*

        - **dataset** : Boolean (default True) - if False and a single data_var,
        return a DataArray
        - **info** : Boolean (default True) - if True return an additional DataGroup with
        metadata and data_arrays
        - **ntv_type** : Boolean (default True) - if True add ntv_type to the name
        """
        return Xdataset.from_xarray(self._obj, **kwargs).to_scipp(**kwargs)

    def to_json(self, **kwargs):
        """Accessor for method `Xdataset.from_xarray.to_json` invoked as
        xr.Dataset.nxr.to_json`.

        *Parameters*

        - **encoded** : Boolean (default False) - json value if False else json text
        - **header** : Boolean (default True) - including 'xdataset' type
        - **notype** : list of Boolean (default list of None) - including data type if False
        - **novalue** : Boolean (default False) - including value if False
        - **noshape** : Boolean (default True) - if True, without shape if dim < 1
        - **format** : list of string (default list of 'full') - representation
        format of the ndarray,

        """
        return Xdataset.from_xarray(self._obj, **kwargs).to_json(**kwargs)
