# Mikes Mice

An implementation of Chained Equations and MICE imputation methods for pandas DataFrames. Unlike other notable imputation libraries (eg fancyimpute), this library has support for imputation of categorical variables, as well as the ability to define the prediction models used.

## Usage
```python
import pandas as pd
from mikes_mice import MikesMICE

# X is the input pandas DataFrame
mice_imputer = MikesMICE()
X_filled = mice_imputer.ChainedEquation(X, ['col1', 'col2'])