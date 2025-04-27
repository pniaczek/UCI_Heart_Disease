import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from src.visualization.dimensionality_reduction import plot_pca_variance, plot_pca_feature_loadings, plot_pca_scatter
from loggers import logger_pipelines as logger_pipe

DIM_REDUCTION_PATH = 'plots/02_dimensionality_reduction'


class PCATransformer:
    def __init__(self, variance_threshold=0.9, random_state=0):
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.pca = None
        self.n_components_optimal = None
        self.feature_names_in_ = None
        self.fitted = False

    def fit(self, X_train, plot_results=True, source_str='train', n_features_plot=10):
        logger_pipe.info(
            f"Fitting PCA transformer with variance threshold: {self.variance_threshold}...")

        self.feature_names_in_ = X_train.columns.tolist()

        try:
            pca_full = PCA(random_state=self.random_state)
            pca_full.fit(X_train)

            variance_ratios = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_ratios)

            n_components_optimal_calc = np.argmax(
                cumulative_variance >= self.variance_threshold) + 1
            self.n_components_optimal = int(
                n_components_optimal_calc)

            logger_pipe.info(
                f"Optimal number of components for {self.variance_threshold*100:.1f}% variance: {self.n_components_optimal}")

            self.pca = PCA(n_components=self.n_components_optimal,
                           random_state=self.random_state)
            self.pca.fit(X_train)
            self.fitted = True
            logger_pipe.info(
                f"PCA transformer fitted with {self.n_components_optimal} components.")

            if plot_results:
                logger_pipe.info(
                    f"Generating PCA variance plot for source: {source_str}")
                os.makedirs(DIM_REDUCTION_PATH, exist_ok=True)
                variance_plot_path = os.path.join(
                    DIM_REDUCTION_PATH, f'pca_variance_{source_str}.png')
                plot_pca_variance(
                    variance_ratios,
                    cumulative_variance,
                    output_path=variance_plot_path
                )
                logger_pipe.info(
                    f"Saved PCA variance plot to {variance_plot_path}")

                for i in range(self.n_components_optimal):
                    logger_pipe.info(
                        f"Generating PCA feature loadings plot for component {i+1} (Source: {source_str})")
                    contribution_plot_path = os.path.join(
                        DIM_REDUCTION_PATH, f'pca_comp_features_{source_str}_dim{i + 1}.png')
                    plot_pca_feature_loadings(
                        pca=self.pca,
                        feature_names=self.feature_names_in_,
                        component_index=i,
                        output_path=contribution_plot_path,
                        n_features=n_features_plot
                    )
                    logger_pipe.info(
                        f"Saved PCA feature loadings plot to {contribution_plot_path}")

        except Exception as e:
            logger_pipe.error(f"Error during PCA fit: {e}")
            self.fitted = False

        return self

    def transform(self, X, y=None, plot_scatter=True, source_str='transform'):
        if X is None or X.empty:
            logger_pipe.error("Input data for PCA transformation is empty.")
            return None

        if list(X.columns) != self.feature_names_in_:
            logger_pipe.warning(
                "Input columns for transform do not match columns used with PCA fit.")
            X_aligned = X[self.feature_names_in_]
        else:
            X_aligned = X

        logger_pipe.info(
            f"Applying PCA transformation. Reducing to {self.n_components_optimal} components.")
        try:
            X_pca = self.pca.transform(X_aligned)
            pca_columns = [
                f'PC{i+1}' for i in range(self.n_components_optimal)]
            df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

            logger_pipe.info(
                f"PCA transformation applied. Output shape: {df_pca.shape}")

            return df_pca

        except Exception as e:
            logger_pipe.error(f"Error during PCA transform: {e}")
            return None
