### *NTV-NumPy : A multidimensional semantic, compact and reversible format for interoperability*

<img src="https://loco-philippe.github.io/ES/ntv_pandas.png" alt="ntv-pandas" align="middle" style="height:80px;">

For more information, see the [user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html) or the [github repository](https://github.com/loco-philippe/ntv-numpy).


# Why a new format for multidimensional data ?

Chaque outil a une structure spécifique pour traiter les données multidimensionnelles avec pour conséquences:

- des interfaces dédiés à chaque outil,
- des données traitées partiellement,
- pas de représentation unifiée des structures de données

The proposed format (NTVmulti) is based on the following principles:

- neutral format available for tabular or multidimensional tools (e.g. Numpy, pandas, xarray, scipp, astropy),
- taking into account a wide variety of data types as defined in [NTV](https://www.ietf.org/archive/id/draft-thomy-json-ntv-02.html) format,
- high interoperability: reversible (lossless round-trip) interface with tabular or multidimensional tools,
- reversible and compact JSON format (including categorical and sparse format),
- Ease of sharing and exchanging multidimensional and tabular data,

## main features

The NTV-Numpy converter uses the NTVmulti format pour:

- fournir des interfaces lossless et reversible avec les principaux outils de traitement des données multidimensionnelles et tabulaires,
- offrir des solutions d'échange et de partage de données avec des formats neutres ou banalisés (e.g. JSON, Numpy).

NTV-NumPy was developped originally in the [json-NTV project](https://github.com/loco-philippe/NTV)

## example

In the example below, a dataset available in JSON is shared with scipp or Xarray.

Data example:

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
                
                        'info': {'path': 'https://github.com/loco-philippe/ntv-numpy/tree/main/example/'}
                }
        }

In [2]: from ntv_numpy import Xdataset

        x_example = Xdataset.read_json(example)
        x_example.info
Out[2]: 
```

The JSON representation is equivalent to the Xdataset entity (Json conversion reversible)

```python
In [3]: x_json = x_example.to_json()
        x_example_json = Xdataset.read_json(x_json)
        x_example_json == x_example
Out[2]: True
```

Xarray interoperability

```python
In [4]: x_xarray = x_example.to_xarray()
        x_xarray
Out[4]: xxxx
```

The interface is lossless and reversible.

```python
In [3]: x_example_xr = Xdataset.from_xarray(x_xarray)
        x_example_xr == x_example_json == x_example
Out[2]: True
```

dddd

```python
In [5]: df_to_json = df.npd.to_json()
        pprint(df_to_json, compact=True, width=120, sort_dicts=False)
Out[5]: {':tab': {'index': [100, 200, 300, 400, 500],
                  'dates::date': ['1964-01-01', '1985-02-05', '2022-01-21', '1964-01-01', '1985-02-05'],
                  'value': [10, 10, 20, 20, 30],
                  'value32::int32': [12, 12, 22, 22, 32],
                  'res': [10, 20, 30, 10, 20],
                  'coord::point': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [3.0, 4.0]],
                  'names::string': ['john', 'eric', 'judith', 'mila', 'hector'],
                  'unique': True}}
```

Reversibility:

```python
In [6]: print(npd.read_json(df_to_json).equals(df))
Out[6]: True
```

Table Schema representation:

```python
In [7]: df_to_table = df.npd.to_json(table=True)
        pprint(df_to_table['data'][0], sort_dicts=False)
Out[7]: {'index': 100,
         'dates': '1964-01-01',
         'value': 10,
         'value32': 12,
         'res': 10,
         'coord': [1.0, 2.0],
         'names': 'john',
         'unique': True}

In [8]: pprint(df_to_table['schema'], sort_dicts=False)
Out[8]: {'fields': [{'name': 'index', 'type': 'integer'},
                    {'name': 'dates', 'type': 'date'},
                    {'name': 'value', 'type': 'integer'},
                    {'name': 'value32', 'type': 'integer', 'format': 'int32'},
                    {'name': 'res', 'type': 'integer'},
                    {'name': 'coord', 'type': 'geopoint', 'format': 'array'},
                    {'name': 'names', 'type': 'string'},
                    {'name': 'unique', 'type': 'boolean'}],
         'primaryKey': ['index'],
         'pandas_version': '1.4.0'}
```

Reversibility:

```python
In [9]: print(npd.read_json(df_to_table).equals(df))
Out[9]: True
```
