Version x.y.z
=============

0.1.0 RC1 (2024-03-xx)
--------------------
- First release candidate
- functions `to_json`, `read_json` and `as_def_type` available
- specific methods available in the class `DataFrameConnec` and `SeriesConnec`
- correspondance between `NTVtype` and `dtype` in `ntv_pandas.ini`
- dtype supported:
    - timedelta64[ns], datetime64[ns]
    - string
    - Floatxx, UIntxx, Intxx, boolean
    - categorical
- NTVtype supported:
    - duration, period
    - datetime, date, time, dat
    - month, year, day, wday, yday, week, hour, minute, second
    - json, string, number, boolean, array, object, null
    - floatxx, uintxx, intxx
    - uri, email, file
    - point, line, polygon, geometry, geojson
    - multipoint, multiline, multipolygon, box, codeolc
    - row, field, tab, ntv