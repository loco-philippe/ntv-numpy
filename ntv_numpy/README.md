# installation

## ntv_numpy package

The `ntv_numpy` package includes

- `numpy_ntv_connector` module
  - child classes of `NTV.json_ntv.ntv.NtvConnector` abstract class:
    - `NarrayConnec` class: 'numpy.ndarray' connector
    - `NdarrayConnec` class: 'ndarray'   connector
    - `XndarrayConnec` class: 'xndarray' connector
    - `XdatasetConnec` class: 'xdataset' connector
- `xconnector` module
  - `PandasConnec` class: interface with `pandas.DataFrame`
  - `XarrayConnec` class: interface with `Xarray`
  - `ScippConnec` class: interface with `scipp`
  - `AstropyNDDataConnec` class: interface with `NDData`
- `xdataset` module
  - `Xdataset` class
  - `XdatasetInterface` class: Xdataset interfaces
  - `XdatasetCategory` class: Category of Xndarray in a Xdataset
- `xndarray` module
  - `Xndarray` class
- `ndarray` module
  - `Ndarray` class
  - `Nutil` class: utility functions
  - `NdarrayError` class
- `data_array` module
  - `Darray` class: abstract class
  - `Dfull` class: full format
  - `Dcomplete` class: complete format
  - `Dutil` class: utility functions
- `ndtype` module
  - `Ndtype` class: child class of `Datatype` with additional attributes
- configuration files:
  - `ntv_numpy.ini` (correspondence between ntv_type and numpy dtype)

## Installation

`ntv_numpy` itself is a pure Python package. maintained on [ntv-numpy github repository](https://github.com/loco-philippe/ntv-numpy).

It can be installed with `pip`.

    pip install ntv_numpy

dependency:

- `json_ntv`: support the NTV format,
- `shapely`: for the location data,
- `numpy`: for conversion data
- `Xarray`, `scipp`, `astropy`, `pandas`: for interface
