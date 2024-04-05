### *NTV-NumPy : A multidimensional semantic, compact and reversible format for interoperability*

<img src="https://loco-philippe.github.io/ES/ntv_pandas.png" alt="ntv-pandas" align="middle" style="height:80px;">

For more information, see the [user guide](https://loco-philippe.github.io/ntv-numpy/docs/user_guide.html) or the [github repository](https://github.com/loco-philippe/ntv-numpy).


# Why a NTV-NumPy converter ?

pandas provide JSON converter but three limitations are present:

- the JSON-pandas converter take into account few data types,
- the JSON-pandas converter is not always reversible (conversion round trip)
- external data types (e.g. TableSchema types) are not included

pandas does not have a tool for analyzing tabular structures and detecting integrity errors

## main features

The NTV-pandas converter uses the [semantic NTV format](https://loco-philippe.github.io/ES/JSON%20semantic%20format%20(JSON-NTV).htm)
to include a large set of data types in a JSON representation.

The converter integrates:

- all the pandas `dtype` and the data-type associated to a JSON representation,
- an always reversible conversion,

NTV-NumPy was developped originally in the [json-NTV project](https://github.com/loco-philippe/NTV)

## converter example

In the example below, a DataFrame with multiple data types is converted to JSON (first to NTV format and then to Table Schema format).

The DataFrame resulting from these JSON conversions are identical to the initial DataFrame (reversibility).

With the existing JSON interface, these conversions are not possible.

Data example:

```python
In [1]: from shapely.geometry import Point
        from datetime import date
        import pandas as pd
        import ntv_pandas as npd

In [2]: data = {'index':        [100, 200, 300, 400, 500],
                'dates::date':  [date(1964,1,1), date(1985,2,5), date(2022,1,21), date(1964,1,1), date(1985,2,5)],
                'value':        [10, 10, 20, 20, 30],
                'value32':      pd.Series([12, 12, 22, 22, 32], dtype='int32'),
                'res':          [10, 20, 30, 10, 20],
                'coord::point': [Point(1,2), Point(3,4), Point(5,6), Point(7,8), Point(3,4)],
                'names':        pd.Series(['john', 'eric', 'judith', 'mila', 'hector'], dtype='string'),
                'unique':       True }

In [3]: df = pd.DataFrame(data).set_index('index')
        df.index.name = None

In [4]: df
Out[4]:       dates::date  value  value32  res coord::point   names  unique
        100    1964-01-01     10       12   10  POINT (1 2)    john    True
        200    1985-02-05     10       12   20  POINT (3 4)    eric    True
        300    2022-01-21     20       22   30  POINT (5 6)  judith    True
        400    1964-01-01     20       22   10  POINT (7 8)    mila    True
        500    1985-02-05     30       32   20  POINT (3 4)  hector    True
```

JSON-NTV representation:

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
