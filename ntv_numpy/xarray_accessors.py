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
    del xr.Dataset.nnp
except AttributeError:
    pass

@xr.register_dataset_accessor("nnp")
class NnpDatasetAccessor:
    """Accessor class for methods invoked as `xr.Dataset.nnp.*`"""
    
    def __init__(self, xarray_obj):
        '''initialisation of the class'''
        self._obj = xarray_obj

    def to_dataframe(self, **kwargs):
        """Accessor for method `Xdataset.from_xarray.to_dataframe` invoked as
        xr.Dataset.nnp.to_dataframe`.

        *Parameters*

        - **ntv_type**: Boolean (default True) - if False use full_name else json_name
        - **info**: Boolean (default True) - if True add xdt.info in DataFrame.attrs
        - **dims**: list of string (default None) - order of dimensions full_name to apply
        - **index**: Boolean (default True) - if True, dimensions are translated into indexes
        """
        return Xdataset.from_xarray(self._obj, **kwargs).to_dataframe(**kwargs)

    def to_scipp(self, **kwargs):
        """Accessor for method `Xdataset.from_xarray.to_scipp` invoked as
        xr.Dataset.nnp.to_scipp`.

        *Parameters*

        - **ntv_type**: Boolean (default True) - if False use full_name else json_name
        - **info**: Boolean (default True) - if True add xdt.info in DataFrame.attrs
        - **dims**: list of string (default None) - order of dimensions full_name to apply
        """
        return Xdataset.from_xarray(self._obj, **kwargs).to_scipp(**kwargs)
