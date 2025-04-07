import pandas as pd
from sklearn.impute import KNNImputer
from loggers import logger_imputer as logger
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class Imputer:
    def __init__(self, config):
        self.config = config
        self.strategies = {
            'mean': self._mean_imputation,
            'median': self._median_imputation,
            'mode': self._mode_imputation,
            'knn': self._knn_imputation,
        
        }

    def impute(self, df, column, strategy=None):
        """Imputing missing values in a specific column and with a defined strategy."""
        if df[column].isna().sum() == 0:
            return df

        if strategy is None:
            column_config = self.config['columns'].get(column, {})
            strategy = column_config.get('imputation', 'mean')
        
        if strategy =="mice":
            logger.warning(
                f"Strategy 'mice' is only available as a global imputation method in 'impute_all'."
                f" Skipping column '{column}'..."
        )
            return df

        if strategy not in self.strategies:
            logger.warning(
                f"Unknown imputation strategy '{strategy}' for column '{column}'. Changing to mean strategy instead.")
            strategy = 'mean'

        logger.info(f"Imputing column '{column}' using '{strategy}' strategy")
        return self.strategies[strategy](df, column)

    def impute_all(self, df):
        """Imputing all columns with missing data in the dataframe based on defined strategies."""
        df_copy = df.copy()

        missing_cols = [
            col for col in df_copy.columns if df_copy[col].isna().sum() > 0
        ]

        if missing_cols:
            logger.info(
                f"Found {len(missing_cols)} columns with missing values: {missing_cols}")

            for col in missing_cols:
                strategy = self.config['columns'].get(col, {}).get('imputation', 'mean')
                if strategy != 'mice':
                    df_copy = self.impute(df_copy, col, strategy)
            if df_copy.isna().sum().sum() > 0:
                df_copy = self._mice_imputation_global(df_copy)
        else:
            logger.info("No missing values found in the dataset")

        return df_copy

    def _mean_imputation(self, df, column):
        df_copy = df.copy()
        mean_value = df_copy[column].mean()
        df_copy[column] = df_copy[column].fillna(mean_value)
        logger.info(f"Imputed '{column}' with mean value: {mean_value}")
        return df_copy

    def _median_imputation(self, df, column):
        df_copy = df.copy()
        median_value = df_copy[column].median()
        df_copy[column] = df_copy[column].fillna(median_value)
        logger.info(f"Imputed '{column}' with median value: {median_value}")
        return df_copy

    def _mode_imputation(self, df, column):
        df_copy = df.copy()
        mode_value = df_copy[column].mode()[0]
        df_copy[column] = df_copy[column].fillna(mode_value)
        logger.info(f"Imputed '{column}' with mode value: {mode_value}")
        return df_copy

    def _knn_imputation(self, df, column, n_neighbors=5):
        df_copy = df.copy()

        numeric_columns = df_copy.select_dtypes(
            include=['float64', 'int64']).columns.tolist()

        if column not in numeric_columns:
            logger.warning(
                f"KNN imputation not suitable for non-numeric float64 or int64 column '{column}'. Using mode instead."
            )
            return self._mode_imputation(df_copy, column)

        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_numeric = df_copy[numeric_columns]
        imputed_data = imputer.fit_transform(df_numeric)

        column_idx = numeric_columns.index(column)
        df_copy[column] = imputed_data[:, column_idx]

        logger.info(
            f"Imputed '{column}' using KNN imputation with {n_neighbors} neighbors"
        )

        return df_copy
    
    def _mice_imputation_global(self, df, max_iter=10):
        """Perform global MICE imputation on the entire DataFrame ."""
        df_copy = df.copy()
        logger.info("Applying global MICE (Iterative Imputer) to all numerical columns with missing values...")
        imputer = IterativeImputer(max_iter=max_iter, random_state=0)
        imputed_data = imputer.fit_transform(df_copy)
        df_imputed = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
        logger.info("Global MICE imputing completed. Rounding all values to nearest integers...")
        df_imputed = df_imputed.round()

        # Clip categorical columns to allowed ranges
        for col in df_imputed.columns:
            col_config = self.config['columns'].get(col, {})
            col_type = col_config.get('type')

            if col_type == 'categorical':
                mapping = col_config.get('mapping', {})
                if mapping:
                    allowed_vals = list(mapping.keys())
                    min_val, max_val = min(allowed_vals), max(allowed_vals)
                    df_imputed[col] = df_imputed[col].clip(lower=min_val, upper=max_val)
                    logger.info(f"Clipped values in categorical column '{col}' to range [{min_val}, {max_val}]")

        return df_imputed
    
    




