# Authors: Michael Tremeer <contact@michaeltremeer.com>

from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning
import warnings
import numpy as np
import pandas as pd
import time
import gc

class ChainedImputer():
    def __init__(self, cat_cols_for_impute, cont_cols_for_impute, cat_model='log reg', cont_model='lin reg'):
        """
        Encode categorical and continuous variables using Chained Equations and Multiple Imputation
        Chained Equations (MICE) methods. Supports pandas DataFrames with familiar fit/fit_transform/transform
        methods.

        Parameters
        ----------
        cat_cols_for_impute : list of string objects
            List of categorical column names to be used as predictors in imputation process and to be imputed
            (if missing values are present). Columns must be of number datatypes (no support yet for strings).
            Columns defined must also be present in 

        cont_cols_for_impute : list of strong objects
            List of continuous column names to be used as predictors in imputation process and to be imputed
            (if missing values are present). Columns must be of number datatypes.

        n_iter : int, default 10
            Number of iterations of of Chained Equations to complete. Each column will be imputed n_iter
            times in total.

        inplace : bool, default True
            If True, do operation inplace and return None.
            
        print_changes : bool, default True
            If True, print percentage change in imputed categorical and continuous variables at the
            end of each iteration.

        Returns
        -------
        self
        """
        # Set models
        self.cat_model = LogisticRegression() if cat_model == 'log reg' else cat_model
        self.cont_model = LinearRegression() if cont_model == 'lin reg' else cont_model
        self.cat_cols_for_impute = cat_cols_for_impute
        self.cont_cols_for_impute = cont_cols_for_impute
        self.fit_completed = False
        self.transform_mode = False

        # Create dict of estimators
        self.estimators_ = {}
        for col in self.cat_cols_for_impute:
            self.estimators_[col] = clone(self.cat_model)
        for col in self.cont_cols_for_impute:
            self.estimators_[col] = clone(self.cont_model)

    @staticmethod
    def _get_cols_for_impute(df):
        """ Returns list of columns for imputation, sorted by % missing (desc)"""
        percent_missing = (df.isnull().sum() / len(df))
        return percent_missing[percent_missing > 0].sort_values(ascending=False).index.tolist()

    @staticmethod
    def _get_mask(df):
        """ Return mask of missing values"""
        return pd.isnull(df)

    def _seed_values(self, df, cat_strategy="most_frequent", cont_strategy="mean"):
        """Performs simple imputation on a DataFrame. cat_cols_for_impute and cont_cols_for_impute
        should be lists of variables to be imputed."""
        if not self.transform_mode:
            # Create imputers
            self.cat_imputer = Imputer(strategy=cat_strategy)
            self.cont_imputer = Imputer(strategy=cont_strategy)
            # Fit imputers
            self.cat_imputer.fit(df[self.cat_cols_for_impute])
            self.cont_imputer.fit(df[self.cont_cols_for_impute])
        # Return imputed df
        return pd.concat([pd.DataFrame(self.cat_imputer.transform(df[self.cat_cols_for_impute]),
                            columns=self.cat_cols_for_impute, index=df.index),
                         pd.DataFrame(self.cont_imputer.transform(df[self.cont_cols_for_impute]),
                            columns=self.cont_cols_for_impute, index=df.index)],
                         axis=1,
                         ignore_index=False)

    def _fill_col(self, train_df, estimator, col_for_prediction, mask_indices,):
        """Takes one_hot_encoded df as input, fits model on all rows, predicts target col on rows with
        missing values, and returns the column with new predicted values"""
        
        # If in fit or fit_transform mode, fit model on rows where original values exist
        if not self.transform_mode: estimator.fit(train_df, col_for_prediction)
        # Predict on missing rows
        predictions = estimator.predict(train_df.iloc[mask_indices])
        # Return column with predictions
        predicted_col = col_for_prediction.copy()
        predicted_col.iloc[mask_indices] = predictions.reshape(-1,1)
        return predicted_col

    @staticmethod
    def _reverse_one_hot(df, one_hot_dict): #tested
        """Collapses one-hot encoded variables into their original columns"""
        for original_column_name in one_hot_dict:
            one_hot_columns = one_hot_dict[original_column_name]
            single_column = df[one_hot_columns].idxmax(axis=1).apply(lambda a: float(a.split('__sep__')[1]))
            df.drop(columns=one_hot_columns, inplace=True)
            df[original_column_name] = single_column

    def fit(self, df, n_iter=10, verbose=0):
        """
        Fits Imputer on DataFrame by Performing n_iter iterations of Chained Equation imputation and fitting
        estimators on the resulting DataFrame. If DataFrame has no missing data, estimators are fit on the entire dataset.

        Parameters
        ----------
        df : pandas DataFrame
        The DataFrame to be imputed. Any columns not defined in cat_cols_for_impute or cont_cols_for_impute
        will be dropped.

        n_iter : int, default 10
        Number of iterations of of Chained Equations to complete. Each column will be imputed n_iter
        times in total. The higher this number, the closer to a minima the imputed values will become.

        verbose : int (default = 0)
        Use verbose = 1 to have updates printed at the end of each iterations, and verbose = 2 to print information
        on the percentage change in imputed variables at the end of each iteration.

        Returns
        -------
        None
        """
        if verbose>0: print('Beginning fit process.')
        self.transform_mode = False
        _ = self.fit_transform(df, n_iter=n_iter, inplace=False, verbose=verbose)#, refit_after_transform=True)
        if verbose>0: print('Fitting complete.')
        return None

    def transform(self, df, n_iter=10, inplace=True, verbose=1):
        """
        Imputes missing values in DataFrame by Performing n_iter iterations of Chained Equation imputation. If DataFrame has no
        missing data, original data is returned.

        Parameters
        ----------
        df : pandas DataFrame
        The DataFrame to be imputed. Any columns not defined in cat_cols_for_impute or cont_cols_for_impute
        will be dropped.

        n_iter : int, default 10
        Number of iterations of of Chained Equations to complete. Each column will be imputed n_iter
        times in total. The higher this number, the closer to a minima the imputed values will become.

        verbose : int (default = 0)
        Use verbose = 1 to have updates printed at the end of each iterations, and verbose = 2 to print information
        on the percentage change in imputed variables at the end of each iteration.

        Returns
        -------
        imputed: pandas.DataFrame
        """
        # Set transform_mode to true so no imputers and estimators are changed from training
        self.transform_mode = True
        if inplace:
            self.fit_transform(df, n_iter=n_iter, inplace=True, verbose=verbose, refit_after_transform=False)
            self.transform_mode = False
            return None
        else:
            transformed_df = self.fit_transform(df, n_iter=n_iter, inplace=False, verbose=verbose, refit_after_transform=False)
            self.transform_mode = False
            return transformed_df

    def fit_transform(self, df, n_iter=10,  inplace=True, verbose=1, refit_after_transform=True):
        """
        Performs n_iter iterations of Chained Equation imputation on dataframe, and then fits Imputer on the imputed dataframe.
        ``cat_cols_for_impute`` and ``cont_cols_for_impute`` are lists of categorical and continuous columns to use as predictor
        & imputation targets. If DataFrame has no missing data, estimators are fit on the entire dataset.

        Parameters
        ----------
        df : pandas DataFrame
        The DataFrame to be imputed. Any columns not defined in cat_cols_for_impute or cont_cols_for_impute
        will be dropped.

        n_iter : int, default 10
        Number of iterations of of Chained Equations to complete. Each column will be imputed n_iter
        times in total. The higher this number, the closer to a minima the imputed values will become.

        inplace : bool, default True
            If True, do operation inplace and return None.

        verbose : int (default = 0)
        Use verbose = 1 to have updates printed at the end of each iterations, and verbose = 2 to print information
        on the percentage change in imputed variables at the end of each iteration.

        refit_after_transform : bool, default True
        If True, will refit all estimators on final imputed DataFrame after imputation is complete. Recommended if
        imputer will be reused on test data.

        Returns
        -------
        imputed: pandas.DataFrame
        """
        if verbose > 0: print('Seeding df and one hot encoding...')
        # Find cols for impute
        cols_for_impute = self._get_cols_for_impute(df[self.cat_cols_for_impute+self.cont_cols_for_impute])
        # Store number of missing values
        total_missing = df.isna().sum().sum()
        # Create mask of missing values with continuous cols before categorical cols
        mask = self._get_mask(df[self.cont_cols_for_impute+self.cat_cols_for_impute])
        # Save original columns to list to restore order after transform
        self.original_cols = df.columns.tolist()
        # Create seeded df to train on
        seeded_df = self._seed_values(df)[self.cont_cols_for_impute+self.cat_cols_for_impute]
        if not self.transform_mode:
            # Create OneHotEncoder and fit on seeded df
            self.cat_cols_for_impute_idx = [seeded_df.columns.get_loc(c) for c in seeded_df.columns if c in self.cat_cols_for_impute]
            self.one_hot_enc = OneHotEncoder(categorical_features=self.cat_cols_for_impute_idx, sparse=False)
            self.one_hot_enc.fit(seeded_df)
            # Get one_hot_encoded col_names for stitching OneHotEncoder output back to df
            self.one_hot_cols = pd.get_dummies(seeded_df, prefix_sep='__sep__', columns=self.cat_cols_for_impute).drop(columns=self.cont_cols_for_impute).columns.tolist()
        # Use one_hot_enc to one hot encode categorical vars and to with continuous columns
        one_hot_seeded = pd.concat([seeded_df[self.cont_cols_for_impute], pd.DataFrame(self.one_hot_enc.transform(seeded_df)[:,:len(self.one_hot_cols)], columns=self.one_hot_cols, index=seeded_df.index)], axis=1, ignore_index=False)
        # Free memory
        del seeded_df
        gc.collect()

        # Create one_hot_dict for categorical variables
        if not self.transform_mode:
            self.one_hot_dict = {}
            for col in one_hot_seeded.columns.tolist():
                if len(col.split('__sep__')) > 1:
                    if col.split('__sep__')[0] in self.one_hot_dict:
                        self.one_hot_dict[col.split('__sep__')[0]].append(col)
                    else:
                        self.one_hot_dict[col.split('__sep__')[0]] = [col]

        # Complete n_iter rounds of imputation if some missing values present
        if total_missing > 0:
            for n in range(n_iter):
                if verbose > 0: print(f'Starting iteration {n+1}...')
                iteration_cat_change = []
                iteration_cont_change = []
                # Record time of iteration start
                iter_start_time = time.time()
                # Complete one round of imputation using fill_col()
                for i, col in enumerate(cols_for_impute):
                    if verbose > 2: print(f'Imputing {col} ({i+1}/{len(cols_for_impute)})...')
                    # Set correct model
                    estimator = self.estimators_[col]
                    # Get list of columns to drop from prediction df and if categorical, create column in its original state
                    if col in self.cat_cols_for_impute:
                        cols_of_interest = self.one_hot_dict[col] # drop these cols before feeding to fill_col()
                        col_for_prediction = one_hot_seeded[cols_of_interest].idxmax(axis=1) # use this as target column
                    else:
                        cols_of_interest = [col]
                        col_for_prediction = one_hot_seeded[cols_of_interest].copy() # use this as target column
                    # Get mask indices where values are missing
                    mask_indices = np.where(mask[col])[0]
                    # Pass one-hot-encoded seeded_df and target column to fill_col() and update seeded_df with result
                    if col in self.cat_cols_for_impute:
                        filled_col = self._fill_col(one_hot_seeded.drop(columns=cols_of_interest), estimator, 
                                                    col_for_prediction, mask_indices)
                        # Find percentage of values changes
                        cat_change = (1 - sum(np.equal(filled_col.iloc[mask_indices],
                            one_hot_seeded[cols_of_interest].idxmax(axis=1).loc[mask_indices])) / len(mask_indices)) * 100
                        # Save percentage change to list
                        iteration_cat_change.append(cat_change)
                        # Update df with new values
                        one_hot_seeded[self.one_hot_dict[col]] = pd.get_dummies(filled_col, prefix_sep='__sep__', columns=self.one_hot_dict[col])
                    if col in self.cont_cols_for_impute:
                        # Ignore DataConversionWarnings since sklearn is returning unfixable warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings(action='ignore', category=DataConversionWarning)
                            filled_col = self._fill_col(one_hot_seeded.drop(columns=cols_of_interest, axis=1), estimator, col_for_prediction, mask_indices)
                            # Find percentage of values changes
                            cont_change = 100 * (abs(filled_col.iloc[mask_indices] - one_hot_seeded[cols_of_interest].iloc[mask_indices])).sum() / one_hot_seeded[cols_of_interest].iloc[mask_indices].sum()
                            # Save percentage change to list              
                            iteration_cont_change.append(cont_change)
                            # Update df with new values
                            one_hot_seeded[col] = filled_col
                if verbose > 1:
                    print(f'--> Mean percentage change in imputed continuous values: %.2f%%.\n--> Mean percentage change in imputed categorical values: %.2f%%\nIteration %d completed in %.2f seconds.\n' % 
                        (np.mean(iteration_cont_change), np.mean(iteration_cat_change), 
                            n+1, time.time() - iter_start_time))
                if verbose == 1: print(f'Iteration %d completed in %.2f seconds.' % (n+1, time.time() - iter_start_time))

        if (not self.transform_mode) and refit_after_transform:
            if verbose > 0: print('Refitting estimators on final dataset...')
            for n, col in enumerate(self.cat_cols_for_impute+self.cont_cols_for_impute):
                # Retrive estimator for chosen column
                estimator = self.estimators_[col]
                # Get list of columns to drop from prediction df and if categorical, create column in its original state
                if col in self.cat_cols_for_impute:
                    cols_of_interest = self.one_hot_dict[col] # drop these cols before feeding to fill_col()
                    col_for_prediction = one_hot_seeded[cols_of_interest].idxmax(axis=1) # use this as target column
                else:
                    cols_of_interest = [col]
                    col_for_prediction = one_hot_seeded[cols_of_interest].copy() # use this as target column
                # Train model
                estimator.fit(one_hot_seeded.drop(cols_of_interest, axis=1), col_for_prediction)
                if verbose > 1: print(f'Estimator fit for {col} ({n+1}/{len(self.cat_cols_for_impute+self.cont_cols_for_impute)})')
            self.fit_completed = True
            if verbose > 0: print('Refitting complete')

        # Reverse one-hot encoding and return processed df
        self._reverse_one_hot(one_hot_seeded, self.one_hot_dict)

        # Ensure new df retains original index and column order
        one_hot_seeded.index = df.index
        final_cols = self.cat_cols_for_impute+self.cont_cols_for_impute
        ordered_final_cols = [col for col in self.original_cols if col in final_cols]

        # Return df with columns ordered as original
        if inplace:
            df[self.cat_cols_for_impute+self.cont_cols_for_impute] = one_hot_seeded[self.cat_cols_for_impute+self.cont_cols_for_impute]
            # print(df.isna().sum())
            return df
        else:
            # one_hot_seeded = one_hot_seeded[ordered_final_cols]
            # print(one_hot_seeded.isna().sum())
            return one_hot_seeded

class MICEImputer(ChainedImputer):
    """
    Encode categorical and continuous variables using Multiple Imputation Chained Equations (MICE) method.
    Supports pandas DataFrames with familiar fit/fit_transform/transform methods.

    Parameters
    ----------
    cat_cols_for_impute : list of string objects
        List of categorical column names to be used as predictors in imputation process and to be imputed
        (if missing values are present). Columns must be of number datatypes (no support yet for strings).
        Columns defined must also be present in 

    cont_cols_for_impute : list of strong objects
        List of continuous column names to be used as predictors in imputation process and to be imputed
        (if missing values are present). Columns must be of number datatypes.

    cat_model : Estimator object
    The model to be used for prediction of categorical variables.  By default, vanilla Logistic Regression models
    will be used for each categorical variable.

    cont_model : Estimator object
    The model to use as predictors of continuous variables.  By default, vanilla Linear Regression models
    will be used for each continuous variable.

    Returns
    -------
    self
    """
    def __init__(self, cat_cols_for_impute, cont_cols_for_impute, cat_model='log reg', cont_model='lin reg'):
        super().__init__(cat_cols_for_impute, cont_cols_for_impute, cat_model=cat_model, cont_model=cont_model)

    def fit_transform(self, df, n_iter=10, n_datasets=3, inplace=True, verbose=1, refit_after_transform=True):
        """
        Performs n_iter iterations of Chained Equation imputation on dataframe, and then refits estimators on the
        imputed dataframe. Memory usage is the same for all n_datasets <= 2. If DataFrame has no missing data, 
        estimators are fit on the entire dataset.

        Parameters
        ----------
        df : pandas DataFrame
        The DataFrame to be imputed. Any columns not defined in cat_cols_for_impute or cont_cols_for_impute
        will be dropped.

        n_iter : int, default 10
        Number of iterations of of Chained Equations to complete. Each column will be imputed n_iter
        times in total. The higher this number, the closer to a minima the imputed values will become.

        inplace : bool, default True
            If True, do operation inplace and return None.

        verbose : int (default = 0)
        Use verbose = 1 to have updates printed at the end of each iterations, and verbose = 2 to print information
        on the percentage change in imputed variables at the end of each iteration.

        refit_after_transform : bool, default True
        If True, will refit all estimators on final imputed DataFrame after imputation is complete. Recommended if
        imputer will be reused on test data.

        Returns
        -------
        imputed: pandas.DataFrame
        """
        print('Starting MICE imputation.')
        if not inplace:
            df = df.copy()
        if verbose > 0: print('Imputing dataset 1:')
        datasets_start_time = time.time()
        datasets = []
        # Only run imputation if no missing values
        if df.isna().sum().sum() > 0:
            # fit_transform once, get back new imputed df
            imputed_df = super().fit_transform(df, n_iter=n_iter, verbose=verbose, inplace=False, refit_after_transform=False)
            datasets.append(imputed_df)
            if verbose > 0: print('Dataset 1 imputed in %2.f seconds.' % (time.time() - datasets_start_time))
            # Create n-1 imputed datasets, updating imputed df with new values
            for n in range(n_datasets - 1):
                print(f'Imputing dataset {n+2}:')
                n_dataset_start_time = time.time()
                extra_imputed_df = super().fit_transform(df, n_iter=n_iter, verbose=verbose, inplace=False, refit_after_transform=False)
                imputed_df = imputed_df * ((n+1)/(n+2)) + extra_imputed_df / (n+2)
                time_to_complete = time.time() - n_dataset_start_time
                if verbose > 0: print('Dataset %i imputed in %2.f %s.\n' % (n+2, (time_to_complete if time_to_complete < 100 else time_to_complete/60), ('seconds' if time_to_complete < 100 else 'minutes')))
        if verbose > 0: print("MICE imputation completed in %1.f minutes\n" % ((time.time() - datasets_start_time) / 60))
        
        if refit_after_transform:
            # Refit the estimators on the MICE imputed dataset
            if verbose > 0: print('Refitting estimators on final MICE-imputed dataset...')
            # Create seeded df to train on
            seeded_df = self._seed_values(df)[self.cont_cols_for_impute+self.cat_cols_for_impute]
            # Use one_hot_enc to one hot encode categorical vars and to with continuous columns
            one_hot_seeded = pd.concat([seeded_df[self.cont_cols_for_impute], pd.DataFrame(self.one_hot_enc.transform(seeded_df)[:,:len(self.one_hot_cols)], columns=self.one_hot_cols, index=seeded_df.index)], axis=1, ignore_index=False)
            # Free memory
            del seeded_df
            gc.collect()

            for n, col in enumerate(self.cat_cols_for_impute+self.cont_cols_for_impute):
                # Retrive estimator for chosen column
                estimator = self.estimators_[col]
                # Get list of columns to drop from prediction df and if categorical, create column in its original state
                if col in self.cat_cols_for_impute:
                    cols_of_interest = self.one_hot_dict[col] # drop these cols before feeding to fill_col()
                    col_for_prediction = one_hot_seeded[cols_of_interest].idxmax(axis=1) # use this as target column
                else:
                    cols_of_interest = [col]
                    col_for_prediction = one_hot_seeded[cols_of_interest].copy() # use this as target column
                # Train model
                estimator.fit(one_hot_seeded.drop(cols_of_interest, axis=1), col_for_prediction)
                if verbose > 2: print(f'Estimator fit for {col} ({n+1}/{len(self.cat_cols_for_impute+self.cont_cols_for_impute)})')
            self.fit_completed = True
            if verbose > 0: print('Refitting complete')
        if inplace:
            df[self.cat_cols_for_impute+self.cont_cols_for_impute] = imputed_df[self.cat_cols_for_impute+self.cont_cols_for_impute]
            return None
        return imputed_df

    def transform(self, df, n_iter=10, n_datasets=3, inplace=True, verbose=1):
        """
        Imputer missing values in DataFrame by Performing MICE across n_datasets datasets for n_iter 
        iterations each. Memory usage is the same for all n_datasets <= 2. If DataFrame has no missing 
        data, original data is returned.

        Parameters
        ----------
        df : pandas DataFrame
        The DataFrame to be imputed. Any columns not defined in cat_cols_for_impute or cont_cols_for_impute
        will be dropped.

        n_iter : int, default 10
        Number of iterations of of Chained Equations to complete. Each column will be imputed n_iter
        times in total. The higher this number, the closer to a minima the imputed values will become.

        n_datasets : int, default 10
        Number of datasets to create using MICE. Each dataset will be imputed n_iter times, then combined for the final output.
        The higher this number, the closer to a minima the imputed datasets will become.

        verbose : int, default = 0
        Use verbose = 1 to have updates printed at the end of each iterations, and verbose = 2 to print information
        on the percentage change in imputed variables at the end of each iteration.

        Returns
        -------
        imputed: pandas.DataFrame
        """
        if not self.fit_completed:
            print("Error: Imputer has not yet been fit. Use ``fit`` or ``fit_transform`` first.")
            return df
        self.transform_mode = True
        if inplace:
            self.fit_transform(df, n_iter=n_iter, n_datasets = n_datasets, inplace=True, verbose=verbose, refit_after_transform=False)
            self.transform_mode = False
            return df
        else:
            transformed_df = self.fit_transform(df, n_iter=n_iter, n_datasets = n_datasets, inplace=False, verbose=verbose, refit_after_transform=False)
            self.transform_mode = False
            return transformed_df

    def fit(self, df, n_iter=10, n_datasets=3, verbose=1):
        """
        Fits Imputer on DataFrame by Performing MICE across n_datasets datasets for n_iter iterations each.
        Memory usage is the same for all n_datasets <= 2. ``cat_cols_for_impute``  and ``cont_cols_for_impute`` 
        are lists of categorical and continuous columns to use as predictor & imputation targets. If DataFrame 
        has no missing data, estimators are fit on the entire dataset.

        Parameters
        ----------
        df : pandas DataFrame
        The DataFrame to be imputed. Any columns not defined in cat_cols_for_impute or cont_cols_for_impute
        will be dropped.

        n_iter : int, default 10
        Number of iterations of of Chained Equations to complete. Each column will be imputed n_iter times in total. 
        The higher this number, the closer to a minima the imputed values will become. I recommend 10, but use verbose=1
        to find the optimal number.

        n_datasets : int, default 10
        Number of datasets to create using MICE. Each dataset will be imputed n_iter times, then combined for the final output.
        The higher this number, the closer to a minima the imputed datasets will become. I recommend 3 for fitting.

        verbose : int (default = 1)
        Use verbose = 1 to have updates printed at the end of each iterations, and verbose = 2 to print information
        on the percentage change in imputed variables at the end of each iteration.

        Returns
        -------
        None
        """
        if verbose > 0: print('Beginning fit process. Generating datasets...')
        _ = self.fit_transform(df, n_iter=n_iter, n_datasets=n_datasets, inplace=False, verbose=verbose)
        self.fit_completed = True
        if verbose > 0: print('Fit completed.')
        return None