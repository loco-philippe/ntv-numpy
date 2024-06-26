### *NTV-NumPy : A multidimensional semantic, compact and reversible format for interoperability*

For more information, see the [user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html) or the [github repository](https://github.com/loco-philippe/ntv-numpy).

# Why a new format for multidimensional data ?

Each tool has a specific structure for processing multidimensional data with the following consequences:

- interfaces dedicated to each tool,
- partially processed data,
- no unified representation of data structures

The proposed format is based on the following principles:

- neutral format available for tabular or multidimensional tools (e.g. Numpy, pandas, xarray, scipp, astropy),
- taking into account a wide variety of data types as defined in [NTV](https://www.ietf.org/archive/id/draft-thomy-json-ntv-02.html) format,
- high interoperability: reversible (lossless round-trip) interface with tabular or multidimensional tools,
- reversible and compact JSON format (including categorical and sparse format),
- Ease of sharing and exchanging multidimensional and tabular data,

## main features

The NTV-Numpy converter uses this format to:

- provide lossless and reversible interfaces with multidimensional and tabular data processing tools,
- offer data exchange and sharing solutions with neutral or standardized formats (e.g. JSON, Numpy).

NTV-NumPy was developped originally in the [json-NTV project](https://github.com/loco-philippe/NTV)

## example

In the example below, a dataset available in JSON is shared with scipp or Xarray.

```mermaid
---
title: Example of interoperability
---
flowchart LR
    A[Xarray] <--lossless--> B[Neutral\nXdataset]
    D[Scipp] <--lossless--> B
    C[NDData] <--lossless--> B
    B <--lossless--> E[JSON]
    B <--lossless--> F[DataFrame]
```

### Data example

```python
In [1]: example = {
                'example:xdataset': {
                        'var1': [['float[kg]', [2, 2], [10.1, 0.4, 3.4, 8.2]], ['x', 'y']],
                        'var1.variance': [[[2, 2], [0.1, 0.2, 0.3, 0.4]]],
                        'var1.mask1': [[[True, False]], ['x']],
                        'var1.mask2': [[[2, 2], [True, False, False, True]]],

                        'var2': [['var2.ntv'], ['x', 'y']],

                        'x': [['string', ['23F0AE', '578B98']], {'test': 21}],
                        'y': [['date', ['2021-01-01', '2022-02-02']]],

                        'ranking': [['month', [2, 2], [1, 2, 3, 4]], ['var1']],
                        'z': [['float', [10, 20]], ['x']],
                        'z.uncertainty': [[[0.1, 0.2]]],

                        'z_bis': [[['z1_bis', 'z2_bis']]],

                        'info': {'path': 'https://github.com/loco-philippe/ntv-numpy/tree/main/example/',
                        'location': [['string', ['paris']]]}
                }
        }

In [2]: from ntv_numpy import Xdataset

        x_example = Xdataset.read_json(example)
        x_example.info['structure']
Out[2]: {'name': 'example',
        'xtype': 'group',
        'data_vars': ['var1', 'var2'],
        'data_arrays': ['z_bis'],
        'dimensions': ['x', 'y'],
        'coordinates': ['ranking', 'z'],
        'additionals': ['var1.mask1', 'var1.mask2', 'var1.variance', 'z.uncertainty'],
        'metadata': ['info'],
        'uniques': ['location'],
        'validity': 'undefined',
        'length': 4,
        'width': 13}
```

The JSON representation is equivalent to the Xdataset entity (Json conversion reversible)

```python
In [3]: x_json = x_example.to_json()
        x_example_json = Xdataset.read_json(x_json)
        x_example_json == x_example
Out[3]: True
```

### Xarray interoperability

```python
In [4]: x_xarray = x_example.to_xarray()
        print(x_xarray)
Out[4]: <xarray.Dataset> Size: 202B
        Dimensions:        (x: 2, y: 2)
        Coordinates:
          * x              (x) <U6 48B '23F0AE' '578B98'
          * y              (y) datetime64[ns] 16B 2021-01-01 2022-02-02
            ranking        (x, y) int32 16B 1 2 3 4
            z              (x) float64 16B 10.0 20.0
            location       <U5 20B 'paris'
            var1.mask1     (x) bool 2B True False
            var1.mask2     (x, y) bool 4B True False False True
            var1.variance  (x, y) float64 32B 0.1 0.2 0.3 0.4
            z.uncertainty  (x) float64 16B 0.1 0.2
        Data variables:
            var1           (x, y) float64 32B 10.1 0.4 3.4 8.2
        Attributes:
            info:     {'path': 'https://github.com/loco-philippe/ntv-numpy/tree/main/...
            name:     example
            var2:     [['var2.ntv'], ['x', 'y']]
            z_bis:    [['string', ['z1_bis', 'z2_bis']]]
```

Reversibility:

```python
In [5]: x_example_xr = Xdataset.from_xarray(x_xarray)
        x_example_xr == x_example_json == x_example
Out[5]: True
```

### Pandas interoperability

```python
In [6]: x_dataframe = x_example.to_dataframe()
        print(x_example.to_dataframe(json_name=False))
        print(x_xarray)
Out[6]:
                   ranking     z  z.uncertainty  var1  var1.mask1  var1.mask2  \
x      y
23F0AE 2021-01-01        1  10.0            0.1  10.1        True        True
       2022-02-02        2  10.0            0.1   0.4        True       False
578B98 2021-01-01        3  20.0            0.2   3.4       False       False
       2022-02-02        4  20.0            0.2   8.2       False        True

                   var1.variance location
x      y
23F0AE 2021-01-01            0.1    paris
       2022-02-02            0.2    paris
578B98 2021-01-01            0.3    paris
       2022-02-02            0.4    paris
```

Reversibility:

```python
In [7]: x_example_pd = Xdataset.from_dataframe(x_dataframe)
        x_example_pd == x_example_xr == x_example_json == x_example
Out[7]: True
```

### scipp interoperability

```python
In [8]: x_scipp = x_example.to_scipp()
        print(x_scipp['example'])
Out[8]: <scipp.Dataset>
Dimensions: Sizes[x:string:2, y:date:2, ]
Coordinates:
* ranking:month           int32  [dimensionless]  (x:string, y:date)  [1, 2, 3, 4]
* x:string               string  [dimensionless]  (x:string)  ["23F0AE", "578B98"]
* y:date              datetime64            [ns]  (y:date)  [2021-01-01T00:00:00.000000000, 2022-02-02T00:00:00.000000000]
* z:float               float64  [dimensionless]  (x:string)  [10, 20]
Data:
  var1:float            float64             [kg]  (x:string, y:date)  [10.1, 0.4, 3.4, 8.2]  [0.1, 0.2, 0.3, 0.4]
    Masks:
        mask1:boolean      bool  [dimensionless]  (x:string)  [True, False]
        mask2:boolean      bool  [dimensionless]  (x:string, y:date)  [True, False, False, True]
```

Reversibility:

```python
In [9]: x_example_sc = Xdataset.from_scipp(x_scipp)
        x_example_sc == x_example_pd == x_example_xr == x_example_json == x_example
Out[9]: True
```

### NDData interoperability

```python
In [1]: example = {
                'example:xdataset': {
                        'data': [['float[erg/s]', [1,2,3,4]]],
                        'data.mask': [[[False, False, True, True]]],
                        'data.uncertainty': [['float64[std]', [1.0, 1.414, 1.732, 2.0]]],
                        'meta': {'object': 'fictional data.'},
                        'wcs':  {'WCSAXES': 2, 'CRPIX1': 2048.0, 'CRPIX2': 1024.0, 'PC1_1': 1.2905625619716e-05,
                                'PC1_2': 5.9530912331034e-06, 'PC2_1': 5.0220581265601e-06, 'PC2_2': -1.2644774105568e-05,
                                'CDELT1': 1.0, 'CDELT2': 1.0, 'CUNIT1': 'deg', 'CUNIT2': 'deg', 'CTYPE1': 'RA---TAN',
                                'CTYPE2': 'DEC--TAN', 'CRVAL1': 5.63056810618, 'CRVAL2': -72.05457184279, 'LONPOLE': 180.0,
                                'LATPOLE': -72.05457184279, 'WCSNAME': 'IDC_qbu1641sj', 'MJDREF': 0.0, 'RADESYS': 'ICRS'},
                        'psf': [['float[erg/s]', [1,2,3,4]]]
                }
        }
        n_example = Xdataset.read_json(example)
        n_example.info
Out[1]: {'name': 'example',
        'xtype': 'group',
        'data_arrays': ['data', 'psf'],
        'additionals': ['data.mask', 'data.uncertainty'],
        'metadata': ['meta', 'wcs'],
        'validity': 'valid',
        'width': 6}
```

```python
In [2]: n_nddata = n_example.to_nddata()
        n_nddata
Out[2]: NDData([1., 2., ——, ——], unit='erg / s')
```

Reversibility:

```python
In [3]: n_example_ndd = Xdataset.from_nddata(n_nddata)
        n_example_ndd == n_example
Out[3]: True
```

## URI usage

In the example, only structural data is exchanged with json format.

```python
In [1]: example = {
                'example:xdataset': {
                        'var1': [['float[kg]', [2, 2], 'var1.ntv'], ['x', 'y']],
                        'var1.variance': [[[2, 2], 'var1_variance.ntv']],
                        'var1.mask1': [['var1_mask1.ntv'], ['x']],
                        'var1.mask2': [[[2, 2], 'var1_mask2.ntv']],

                        'var2': [['var2.ntv'], ['x', 'y']],

                        'x': [['x.ntv'], {'test': 21}],
                        'y': [['date', 'y.ntv']],

                        'ranking': [['month', [2, 2], 'ranking.ntv'], ['var1']],
                        'z': [['float', 'z.ntv'], ['x']],
                        'z.uncertainty': [['z_uncertainty.ntv']],

                        'z_bis': [['z_bis.ntv']],

                        'info': {'path': 'https://github.com/loco-philippe/ntv-numpy/tree/main/example/'}
                }
        }
```

The complete example can be rebuild with loading data (path + file name).

```python
In [2]: # simulation of reading files at the indicated "path"
        var1          = np.array([10.1, 0.4, 3.4, 8.2])
        var1_variance = Ndarray([0.1, 0.2, 0.3, 0.4], ntv_type='float')
        var1_mask1    = np.array([True, False])
        var1_mask2    = np.array([True, False, False, True])
        var2          = Ndarray('var2.ntv')
        x             = np.array(['23F0AE', '578B98'])
        y             = np.array(['2021-01-01', '2022-02-02'], dtype='datetime64[D]')
        ranking       = np.array([1, 2, 3, 4])
        z             = np.array([10.0, 20.0])
        z_uncertainty = np.array([0.1, 0.2])
        z_bis         = np.array(['z1_bis', 'z2_bis'])

        array_data = [var1, var1_variance, var1_mask1, var1_mask2, var2, x, y, ranking, z, z_uncertainty, z_bis]

        x_example_mixte_numpy = copy(x_example_mixte)
        for data, xnda in zip(array_data, x_example_mixte_numpy.xnd):
        xnda.set_ndarray(Ndarray(data))

        x_example_mixte_numpy == x_example_mixte_json == x_example_sc == x_example_xr == x_example_json == x_example
Out[2]: True
```
