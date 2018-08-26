# Authors: Michael Tremeer <contact@michaeltremeer.com>

from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time

class MikesMICE:
    def __init__(self, cat_model='log reg', cont_model='lin reg'):
        # Set models
        self.cat_model = LogisticRegression() if cat_model == 'log reg' else cat_model
        self.cont_model = LinearRegression() if cont_model == 'lin reg' else cont_model

    @staticmethod
    def _get_cols_for_impute(df):
        """ Returns list of columns for imputation, sorted by % missing (desc)"""
        percent_missing = (df.isnull().sum() / len(df))
        return percent_missing[percent_missing > 0].sort_values(ascending=False).index.tolist()

    @staticmethod
    def _get_mask(df):
        """ Return mask of missing values"""
        return pd.isnull(df)

    @staticmethod
    def _seed_values(df, cat_cols_for_impute, cont_cols_for_impute, cat_strategy="most_frequent",
        cont_strategy="mean"):
        """Performs simple imputation on a DataFrame. cat_cols_for_impute and cont_cols_for_impute
        should be lists of variables to be imputed."""
        # Create imputers
        cat_imputer = Imputer(strategy=cat_strategy)
        cont_imputer = Imputer(strategy=cont_strategy)
        # Fit imputers
        cat_imputer.fit(df[cat_cols_for_impute])
        cont_imputer.fit(df[cont_cols_for_impute])
        # Return imputed df
        return pd.concat([pd.DataFrame(cat_imputer.transform(df[cat_cols_for_impute]),
                            columns=cat_cols_for_impute, index=df.index),
                         pd.DataFrame(cont_imputer.transform(df[cont_cols_for_impute]),
                            columns=cont_cols_for_impute, index=df.index)],
                         axis=1,
                         ignore_index=False)

    @staticmethod
    def _fill_col(train_df, model, col_for_prediction, mask_indices):
        """Takes one_hot_encoded df as input, fits model on all rows, predicts target col on rows with
        missing values, and returns the column with new predicted values"""
        
        # Fit model on rows where original values exist
        model.fit(train_df, col_for_prediction)
        # Predict on missing rows
        predictions = model.predict(train_df.iloc[mask_indices])
        # Return column with predictions
        predicted_col = col_for_prediction
        predicted_col.iloc[mask_indices] = predictions
        return predicted_col

    @staticmethod
    def _reverse_one_hot(df, one_hot_dict): #tested
        """Collapses one-hot encoded variables into their original columns"""
        for original_column_name in one_hot_dict:
            one_hot_columns = one_hot_dict[original_column_name]
            single_column = df[one_hot_columns].idxmax(axis=1).apply(lambda a: str(a.split('__sep__')[1]))
            df.drop(columns=one_hot_columns, inplace=True)
            df[original_column_name] = single_column

    def ChainedImputer(self, df, cat_cols_for_impute, cont_cols_for_impute, n_iter=10, inplace=True, 
                        print_change=True):
        """
        Performs Chained Equation imputation on missing variables in a DataFrame, where ``cat_cols_for_impute``
        and ``cont_cols_for_impute`` are lists of variables to use as predictor & imputation
        columns, and variables in ignore_cols are ignored as predictors.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame to be imputed. Any columns not defined in cat_cols_for_impute or cont_cols_for_impute
            will be ignored during imputation and retained in output DataFrame.

        cat_cols_for_impute : list of string objects
            List of categorical column names to be used as predictors in imputation process and to be imputed
            (if missing values are present). Columns must be of number datatypes (no support yet for strings).

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
        imputed: pandas.DataFrame
        """

        # Set models
        # cat_model = LogisticRegression() if cat_model == 'log reg' else cat_model
        # cont_model = LinearRegression() if cont_model == 'lin reg' else cont_model
        # Return copy if inplace=False
        # Save original columns to list to restore order after transform
        original_cols = df.columns.tolist()
        # Find cols for impute
        cols_for_impute = self._get_cols_for_impute(df[cat_cols_for_impute+cont_cols_for_impute])
        # Create mask of missing values
        mask = self._get_mask(df)
        # Seed df and one-hot-encode categorical vars
        one_hot_seeded = pd.get_dummies(self._seed_values(df, cat_cols_for_impute, cont_cols_for_impute),
            prefix_sep='__sep__', columns=cat_cols_for_impute)
        # Create dictionary of categorical variables and their one-hot-encoded columns
        one_hot_dict = {}
        for col in one_hot_seeded.columns.tolist():
            if len(col.split('__sep__')) > 1:
                if col.split('__sep__')[0] in one_hot_dict:
                    one_hot_dict[col.split('__sep__')[0]].append(col)
                else:
                    one_hot_dict[col.split('__sep__')[0]] = [col]

        # Complete n_iter rounds of imputation
        for n in range(n_iter):
            print(f'Starting round {n+1}...')
            iteration_cat_change = []
            iteration_cont_change = []
            # Record time of iteration start
            iter_start_time = time.time()
            # Complete one round of imputation using fill_col()
            for col in cols_for_impute:
                # Set correct model
                if col in cat_cols_for_impute:
                    model = self.cat_model
                if col in cont_cols_for_impute:
                    model = self.cont_model
                # Get list of columns to drop from prediction df and if categorical, create column in its original state
                if col in cat_cols_for_impute:
                    cols_of_interest = one_hot_dict[col] # drop these cols before feeding to fill_col()
                    col_for_prediction = one_hot_seeded[cols_of_interest].idxmax(axis=1) # use this as target column
                else:
                    cols_of_interest = [col]
                    col_for_prediction = one_hot_seeded[cols_of_interest].copy() # use this as target column
                # Get mask indices where values are missing
                mask_indices = np.where(mask[col])[0]
                # Pass one-hot-encoded seeded_df and target column to fill_col() and update seeded_df with result
                if col in cat_cols_for_impute:
                    filled_col = self._fill_col(one_hot_seeded.drop(columns=cols_of_interest), model, 
                                                col_for_prediction, mask_indices)
                    # Find percentage of values changes
                    cat_change = (1 - sum(np.equal(filled_col.iloc[mask_indices],
                        one_hot_seeded[cols_of_interest].idxmax(axis=1).loc[mask_indices])) / len(mask_indices)) * 100
                    # Save percentage change to list
                    iteration_cat_change.append(cat_change)
                    # Update df with new values
                    one_hot_seeded[one_hot_dict[col]] = pd.get_dummies(filled_col, prefix_sep='__sep__', columns=one_hot_dict[col])
                if col in cont_cols_for_impute:
                    filled_col = self._fill_col(one_hot_seeded.drop(columns=cols_of_interest), model, col_for_prediction, mask_indices)
                    # Find percentage of values changes
                    cont_change = 100 * (abs(filled_col.iloc[mask_indices] - one_hot_seeded[cols_of_interest].iloc[mask_indices])).sum() / one_hot_seeded[cols_of_interest].iloc[mask_indices].sum()
                    # Save percentage change to list              
                    iteration_cont_change.append(cont_change)
                    # Update df with new values
                    one_hot_seeded[col] = filled_col
            if print_change:
                print(f'--> Mean percentage change in imputed continuous values: %.2f%%.\n--> \
                    Mean percentage change in imputed categorical values: %.2f%%\nRound %d \
                    completed in %.2f seconds.\n' % 
                    (np.mean(iteration_cont_change), np.mean(iteration_cat_change), 
                        n+1, time.time() - iter_start_time))
            else:
                print(f'Round %d completed in %.2f seconds.' % (n+1, time.time() - iter_start_time))

        # Reverse one-hot encoding and return processed df
        self._reverse_one_hot(one_hot_seeded, one_hot_dict)

        # Ensure new df retains original index
        one_hot_seeded.index = df.index
            
        # Return df with columns ordered as original
        if inplace:
            df[cat_cols_for_impute+cont_cols_for_impute] = one_hot_seeded[cat_cols_for_impute+cont_cols_for_impute]
            return None
        else:
            return pd.concat([df.drop(columns=cat_cols+cont_cols, axis=1), one_hot_seeded], axis=1, ignore_index=False)[original_cols]