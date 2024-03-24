## Introduction

This memo is a proposal to implement a neutral format for multidimensional data. 

It follows :

- the definition of the [JSON-NTV](https://www.ietf.org/archive/id/draft-thomy-json-ntv-02.html) format (Named and Typed value) which integrates a notion of type in JSON format (see [JSON-NTV package](https://pypi.org/project/json-ntv/)).
- its variation for tabular data ( [NTV-TAB](https://www.ietf.org/archive/id/draft-thomy-ntv-tab-00.html) specification) and its implementation for pandas ([NTV-pandas package](https://pypi.org/project/ntv-pandas/) available in the [pandas ecosystem](https://pandas.pydata.org/community/ecosystem.html) and the [PDEP12](https://pandas.pydata.org/pdeps/0012-compact-and-reversible-JSON-interface.html))
- analysis of tabular structures to identify multi-dimensional data ([TAB-analysis](https://pypi.org/project/tab-analysis/) package)

## Benefits
The use of this format has the following advantages:

- neutral format available for tabular or multidimensional data (e.g. Numpy, pandas, xarray, scipp),
- Taking into account data types not known to other tools for tabular or multidimensional data,
- High level of Interoperability between tools
- reversible and compact JSON format (lossless round-trip, categorical and sparse format, binary coding structure mixing)
- Ease of sharing multi-dimensional data

## Terminology

- **darray (unidimensional array)** is an ordered collection of 'items'. A 'darray' can be represented with several formats (e.g. simple list, categorical format, sparse format)
- **ndarray (multidimensional array)** is a N-dimensional array of homogeneous data types. A ndarray entity is defined by:   
    - a darray of "items" of the same type (flattened multidimensional data ordered with row_major order)
    - a shape defines the order and the length of indexes
    - a data type
- **xndarray (labelled multidimensional array)** is a ndarray defined by a name. A xndarray entity has optional additional data:
    - add_name : the name of a property (additional name to the ndarray name)
    - ntv_type : a semantic extension of the ndarray data type
    - links : the names of xndarray indexes
    - metadata : Json-object metadata
    
    The ndarray can be included in the xndarray or only represented by a resolvable URI.

    Xndarray data can be:
   
        - *named-array* : ndarray with name (without additional data)
        - *variable* : named-array with named indexes (links).
        - *additional-array* : named-array where name is extended (add_name)
- xdataset (coordinated multidimensional array) is a collection of xndarray. This collection can be interpreted as a simple group or as an interconnected collection (names are used as pointers between xndarray items).
In the context of a xdataset, xndarray data can be:

    - dimension : named-array where his name is present in links of a variable of the xdataset
    - data-array : named-array where his name is not present in links of a variable of the xdataset
    - data-var : variable where links equals the list of dimensions of the xdataset
    - coordinate : variable where links not equals the list of dimensions of the xdataset

 A xdataset is valid if :  
    - included Xndarray are valid
    - a names in links is the names of a xndarray
    - the shape of a variable xndarray is consistent with the shape of Xndarray defined in his links
 
  A xdataset is multidimensional if it is valid and contains more than one data-var.
  
  A xdataset is unidimensional if it is valid and contains a single data-var.
  
  In the other cases, a xdataset is a simple group of xndarray.

Example
- Numpy.array corresponds to ndarray
- Xarray.DataArray, Xarray.Dataset, scipp.DataArray, scipp.DataGroup, scipp.Dataset corresponds to xdataset
     
