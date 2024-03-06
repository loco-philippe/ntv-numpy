# installation

## ntv_pandas package

The `ntv_numpy` package includes

- `numpy_ntv_connector` module
  - functions `read_json` and `to_json` to convert JSON data and NumPy entities
  - function `equals` to extend numpy `equals` method
  - child classes of `NTV.json_ntv.ntv.NtvConnector` abstract class:
    - `NdarrayConnec`: 'ndarray'   connector
    - `XndarrayConnec`: 'xndarray' connector
  - an utility class with static methods : `NpUtil`
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
