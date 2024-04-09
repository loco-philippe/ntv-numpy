Version x.y.z
=============

0.1.0 alpha 1 (2024-04-xx)
--------------------
- First version
- specific methods available in the class `DataFrameConnec` and `SeriesConnec`
- correspondance between `NTVtype` and `dtype` in `ntv_numpy.ini`
- numpy.dtype supported: all except 'complex' and 'timedelta'
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
- interface:
    - astropy.NDData
    - scipp.Dataset
    - Xarray.Dataset, Xarray.Dataarray
    - JSON