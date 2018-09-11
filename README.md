# Mikes Mice

An implementation of Chained Equations and Multiple Imputation Chained Equations (MICE) data imputation methods for pandas DataFrames. Unlike other libraries featuring MICE (eg fancyimpute), this library has support for imputation of categorical variables, separate fit and transform functions (so new data can be transformed using models trained on the training set), as well as the ability to define the prediction models used. The Imputer should be used like any other estimator, with familiar fit, fit_transform, and transform methods.

## Usage
```python
import pandas as pd
from mikes_mice import MICEImputer

# X is the input pandas DataFrame
chained_imputer = ChainedImputer(categorical_variables_list, continuous_variables_list)
X_filled = mice_imputer.ChainedEquation(X, n_iter=10)

# Instantiate the imputer, defining continuous/categorical variables and (optionally) estimators to be used
mice_imputer = MICEImputer(categorical_variables_list, continuous_variables_list, cont_model=RandomForestRegressor())

# Fit and transform your training dataset
mice_imputer.fit_transform(X_train, n_dataset=10, n_iter=10)

# Once imputer has been fit on training data, you can now impute new data
mice_imputer.transform(X_test, n_datasets=3, n_iter=10)

