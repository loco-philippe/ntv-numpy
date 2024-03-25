## Interoperability

Example :
```json
{':xdataset': {
  'var1': [['float[kg]', [2, 2], [10.1, 0.4, 3.4, 8.2]], ['x', 'y']],
  'var1.mask1': [['boolean', [True, False]], ['x']],
  'var1.mask2': [['boolean', [2, 2], [True, False, False, True]]],
  'var1.variance': [['float64', [2, 2], [0.1, 0.2, 0.3, 0.4]]],

  'x': [['base16', ['23F0AE', '578B98']]],
  'y': [['date', ['2021-01-01', '2022-02-02']]],
    
  'ranking': [['month', [2, 2], [1, 2, 3, 4]], ['var1']],

  'z': [['float', [10.0, 20.0]], ['x']],
  'z.variance': [['float64', [0.1, 0.2]]]
}}
```
The first ligne is the NTV representation ( {'NTVname:NTVtype': NTVvalue}).

The other lines are the xndarray included in the xdataset (JsonObjects):
   - 'x' and 'y' are *dimension*
   - 'var1' and 'var2' are  *data_var*
   - 'var1.variance' and 'z.variance' are *data_add*
   - 'var1.mask1' and 'var1.mask2' are *mask*
   - 'ranking' and 'z' are *coordinate*
   - 'z_bis' is *data_array*
   - 'info' is *metadata*
   - 'var1' has a NTVtype with an extension ('kg')
   - 'var2' has a relative ndarray
   - 'x' has metadata