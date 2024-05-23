Version x.y.z
=============

0.2.2 RC 1 (2024-05-23)
------------------------
- add Xarray accessor (xr.Dataset.nxr.*)
- pre-commit integration

0.2.1 RC 1 (2024-05-16)
------------------------
- bug #3 (python 3.10)
- enh #4 (Unification of parameters in to_xarray, to_scipp and to_dataframe)
- add CI processus (Github Workflow)

0.2.0 RC 1 (2024-05-05)
------------------------
- add NTVtype extension (Ndtype class)
- add 'uniques' xndarray
- interface:
    - pandas.DataFrame

0.1.2 alpha 1 (2024-04-10)
---------------------------
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
