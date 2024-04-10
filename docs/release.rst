Version x.y.z
=============

0.1.0 alpha 1 (2024-04-10)
--------------------
- First version
- exchanging via JSON format
- lightweight format (structure: json, arrays: URI) 
- numpy.dtype supported: all except 'complex' and 'timedelta'
- NTVtype supported:
    - datetime, date, time, dat
    - month, year, day, wday, yday, week, hour, minute, second
    - json, string, number, boolean, array, object, null
    - floatxx, uintxx, intxx
    - base16, decimal
    - uri, ipv4, email, file
    - point, line, polygon, geometry, geojson
    - multipoint, multiline, multipolygon, box, codeolc
    - ndarray, xndarray, field, tab, ntv
- interface:
    - astropy.NDData
    - scipp.Dataset
    - Xarray.Dataset, Xarray.Dataarray
    - JSON