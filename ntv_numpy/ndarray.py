# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:59:43 2024

@author: a lab in the Air
"""

import pandas as pd
import numpy as np
from json_ntv import Ntv
#from abc import ABC, abstractmethod
from copy import copy
import json

from json_ntv import NtvConnector   
SeriesConnec = NtvConnector.connector().get('SeriesConnec')
DataFrameConnec = NtvConnector.connector().get('DataFrameConnec')

class Ndarray:
    
    @staticmethod 
    def equals(npself, npother):
        '''return True if all elements are equals and dtype are equal'''
        if not (isinstance(npself, np.ndarray) and isinstance(npother, np.ndarray)):
            return False
        if npself.dtype != npother.dtype:
            return False
        if npself.shape != npother.shape:
            return False
        if len(npself.shape) == 0:
            return True
        if len(npself) != len(npother):
            return False
        if isinstance(npself[0], (np.ndarray, pd.Series, pd.DataFrame)):
            equal = {np.ndarray: Ndarray.equals, 
                     pd.Series: SeriesConnec.equals, 
                     pd.DataFrame: DataFrameConnec.equals}                      
            for a, b in zip(npself, npother): 
                if not equal[type(npself[0])](a, b):
                    return False
            return True
        else:
            return np.array_equal(npself, npother)    
        
    @staticmethod 
    def add_ext(typ, ext):
        '''return extended typ'''
        ext = '['+ ext +']' if ext else ''
        return '' if not typ else typ + ext

    @staticmethod
    def split_typ(typ):
        '''return a tuple with typ and extension'''
        if not isinstance(typ, str):
            return (None, None) 
        spl = typ.split('[', maxsplit=1)
        return (spl[0], None) if len(spl) == 1 else (spl[0], spl[1][:-1])        