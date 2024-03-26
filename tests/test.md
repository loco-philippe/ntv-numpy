## Interoperability

Example :
```json
{'test:xdataset': {
    'var1': ['https://github.com/loco-philippe/ntv-numpy/tree/main/example/ex_ndarray.ntv', ['x', 'y']],    
            
    'var2':          [['float[kg]', [2, 2], [10.1, 0.4, 3.4, 8.2]], ['x', 'y']],
    'var2.variance': [[[2, 2], [0.1, 0.2, 0.3, 0.4]]],
    'var2.mask1':    [[[True, False]], ['x']],
    'var2.mask2':    [[[2, 2], [True, False, False, True]]],

    'x':             [['base16', ['23F0AE', '578B98']]], #, {'test': 21}],
    'x.mask1':       [[[True, False]]],

    'y':             [['date', ['2021-01-01', '2022-02-02']]],

    'ranking':       [['month', [2, 2], [1, 2, 3, 4]], ['var2']],

    'z':             [['float', [10, 20]], ['x']],
    'z.uncertainty': [[[0.1, 0.2]]],

    'z_bis':         [[['z1_bis', 'z2_bis']]],
    
    'info': {'example': 'everything'}
}}
```
The first ligne is the NTV representation ( {'NTVname:NTVtype': NTVvalue}).

The other lines are the xndarray included in the xdataset (JsonObjects):
   - `x` and `y` are *dimension*
   - `var1` and `var2` are  *data_var*
   - 'var2.variance' and 'z.uncertainty' are *data_add*
   - 'var2.mask1' and 'var2.mask2' are *mask*
   - 'ranking' and 'z' are *coordinate*
   - 'z_bis' is *data_array*
   - 'info' is *metadata*
   - 'var1' has a NTVtype with an extension ('kg')
   - 'var2' has a relative ndarray
   - 'x' has metadata