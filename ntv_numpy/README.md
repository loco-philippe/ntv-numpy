# installation

## ntv_numpy package

The `ntv_numpy` package includes

- `numpy_ntv_connector` module
  - child classes of `NTV.json_ntv.ntv.NtvConnector` abstract class:
    - `NarrayConnec`: 'np.ndarray'   connector
    - `NdarrayConnec`: 'ndarray'   connector
    - `XndarrayConnec`: 'xndarray' connector
    - `XdatasetConnec`: 'xdataset' connector
- `xconnector` module
  - `XarrayConnec`: interface with `Xarray`
  - `ScippConnec`: interface with `scipp`
  - `AstropyNDDataConnec`: interface with `NDData`
- `xdataset` module
- configuration files:
  - `ntv_numpy.ini` (correspondence between ntv_type and pandas dtype)

## Installation

`ntv_numpy` itself is a pure Python package. maintained on [ntv-numpy github repository](https://github.com/loco-philippe/ntv-numpy).

It can be installed with `pip`.

    pip install ntv_numpy

dependency:

- `json_ntv`: support the NTV format,
- `shapely`: for the location data,
- `numpy`
