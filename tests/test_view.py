# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 07:18:18 2024

@author: phili
"""
import pandas as pd
import ntv_pandas as npd
from tab_analysis import Util
fruits = {'plants':      ['fruit', 'fruit', 'fruit', 'fruit', 'vegetable', 'vegetable', 'vegetable', 'vegetable'],
  'plts':        ['fr', 'fr', 'fr', 'fr', 've', 've', 've', 've'], 
  'quantity':    ['1 kg', '10 kg', '1 kg', '10 kg', '1 kg', '10 kg', '1 kg', '10 kg'],
  'product':     ['apple', 'apple', 'orange', 'orange', 'peppers', 'peppers', 'carrot', 'carrot'],
  'price':       [1, 10, 2, 20, 1.5, 15, 1.5, 20],
  'price level': ['low', 'low', 'high', 'high', 'low', 'low', 'high', 'high'],
  'group':       ['fruit 1', 'fruit 10', 'fruit 1', 'veget', 'veget', 'veget', 'veget', 'veget'],
  'id':          [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
  'supplier':    ["sup1", "sup1", "sup1", "sup2", "sup2", "sup2", "sup2", "sup1"],
  'location':    ["fr", "gb", "es", "ch", "gb", "fr", "es", "ch"],
  'valid':       ["ok", "ok", "ok", "ok", "ok", "ok", "ok", "ok"]}
simple = { 'a': [1,2,3,4,4],
           'b': [10,20,30,40,40],
           'c': [1,1,3,4,4],
           'd': [1,1,1,4,4],
           'e': [1,1,1,1,1]}
df1 = pd.DataFrame(simple)
a_df = df1.npd.analysis(distr=True)
print(a_df.partitions(mode='id'))
print(a_df.field_partition(mode='id'))
print(a_df.relation_partition())
"""
df1 = pd.DataFrame(fruits)
a_df = df1.npd.analysis(distr=True)

print(a_df.field_partition(partition=['plants', 'price level', 'quantity'], mode='id'))
print(a_df.field_partition(partition=['plts', 'price level', 'quantity'], mode='id'))
print(a_df.relation_partition(partition=['plants', 'price level', 'quantity']))
'''print(Util.view(a_df.partitions()[0], mode='id'))
print(Util.view(a_df.partitions()[0], mode='index', ana=a_df))
print(Util.view(a_df.partitions()[0], mode='field', ana=a_df))

print(Util.view(a_df.partitions(mode='id')[0], mode='id', ana=a_df))
print(Util.view(a_df.partitions(mode='id')[0], mode='index', ana=a_df))
print(Util.view(a_df.partitions(mode='id')[0], mode='field', ana=a_df))
'''
"""