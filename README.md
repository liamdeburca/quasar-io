# Quasar IO

A Python package providing input/output utilities for the QUASAR framework. 

Currently, this package supports the following formats, each intended for different types of data and use cases.

1. JSON (.json): General serialization format for structured data. Can optionally write 
to other formats, such as YAML (.yaml, .yml), TOML (.toml), or TOON (.toon). 
2. HDF5 (.h5, .hdf5): Hierarchical Data Format, suitable for storing large 
amounts of numerical data, such as arrays, efficiently.