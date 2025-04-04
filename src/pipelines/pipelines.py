import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from src.data_processing.processor import Processor
from src.data_processing.data_transformation import DataTransformer
from src.visualization.missing_values import plot_missing_values_bar_plot, plot_missing_values_matrix_plot, plot_missing_values_heatmap
from src.visualization.distributions import plot_all_distributions
from src.visualization.correlation import plot_data_correlation
from src.visualization.dimensionality_reduction import plot_pca_variance, plot_pca_feature_loadings, plot_pca_scatter
from loggers import logger_main as logger
from loggers import logger_pipelines as logger_pipe


CONFIG_PATH = 'config/data_config.yaml'
PROCESSED_DATA_PATH = 'data/processed'
EDA_PATH = 'plots/01_basic_eda'
DIM_REDUCTION_PATH = 'plots/02_dimensionality_reduction'


def missing_data_visualization_pipeline(source_key=None):
    source_str = 'uci' if source_key is None else source_key

    logger.info(
        "Starting data processing and visualization pipeline for missing values..."
    )
    processor = Processor(CONFIG_PATH, PROCESSED_DATA_PATH)

    logger.info(f"Processing {source_str} dataset with missing values...")
    df_missing_values = processor.process_data(
        impute=False,
        source_key=source_key,
        save_csv_source_key=f'{source_str}_missing_values'
    )

    if df_missing_values is not None:
        logger.info(
            f"{source_str} dataset with missing values processed successfully."
        )
        os.makedirs(EDA_PATH, exist_ok=True)

        plot_missing_values_bar_plot(
            df_missing_values,
            output_dir=EDA_PATH,
            filename=f'{source_str}_missing_values_bar_plot.png'
        )

        plot_missing_values_matrix_plot(
            df_missing_values,
            output_dir=EDA_PATH,
            filename=f'{source_str}_missing_values_matrix_plot.png'
        )

        plot_missing_values_heatmap(
            df_missing_values,
            output_dir=EDA_PATH,
            filename=f'{source_str}_missing_values_heatmap.png'
        )
    else:
        logger.error(f"Failed to process {source_str} dataset.")

    logger.info(
        "Data processing and visualization pipeline for missing values finished."
    )


def processing_pipeline(source_key=None, feature_engineering=True, visualizations=True):
    source_str = 'uci' if source_key is None else source_key

    logger.info("Starting processing pipeline...")
    processor = Processor(CONFIG_PATH, PROCESSED_DATA_PATH)

    logger.info(f"Processing {source_str} dataset...")

    if feature_engineering:
        df_processed = processor.add_feature_engineering(
            source_key=source_key,
            impute=True,
            source_str=source_str
        )
    else:
        df_processed = processor.process_data(
            source_key=source_key,
            impute=True,
            save_csv_source_key=source_str
        )

    if df_processed is not None:
        logger.info(f"{source_str} dataset processed successfully.")
    else:
        logger.error(f"Failed to process {source_str} dataset.")
        return df_processed

    if visualizations:
        os.makedirs(EDA_PATH, exist_ok=True)

        plot_all_distributions(
            df_processed,
            output_dir=EDA_PATH
        )
        plot_data_correlation(
            df_processed,
            output_dir=EDA_PATH
        )

    logger.info("Data processing pipeline finished.")
    return df_processed


def transformation_pipeline(df_processed, source_key=None):
    source_str = 'uci' if source_key is None else source_key

    logger_pipe.info(
        f"Starting transformation pipeline for {source_str} dataset...")
    transformer = DataTransformer()

    try:
        df_transformed = transformer.fit_transform(df_processed)
        logger_pipe.info("Data transformation applied successfully.")

        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        output_filename = f'data_transformed_{source_str}.csv'
        output_path = os.path.join(PROCESSED_DATA_PATH, output_filename)
        df_transformed.to_csv(output_path, index=False)
        logger_pipe.info(f"Saved transformed data to {output_path}")

    except Exception as e:
        logger_pipe.error(f"Error during transformation pipeline: {e}")
        return None

    logger_pipe.info(f"Transformation pipeline finished.")
    return df_transformed


def dim_reduction_pipeline(df_transformed, source_key=None, target_col='target', variance_threshold=0.9, n_features_plot=10):
    source_str = 'uci' if source_key is None else source_key

    if df_transformed is None or df_transformed.empty:
        logger_pipe.error(
            "Input data for dimensionality reduction is empty.")
        return None

    logger_pipe.info(
        f"Starting dimensionality reduction pipeline for {source_str}...")

    try:
        if target_col not in df_transformed.columns:
            logger_pipe.error(
                f"Target column '{target_col}' not found in the data.")
            return None

        X = df_transformed.drop(columns=[target_col])
        y = df_transformed[target_col]
        feature_names = X.columns.tolist()

        pca_full = PCA(random_state=0)
        pca_full.fit(X)

        variance_ratios = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratios)

        n_components_optimal = np.argmax(
            cumulative_variance >= variance_threshold) + 1

        logger_pipe.info(
            f"Optimal number of components for {variance_threshold*100}% variance: {n_components_optimal}")

        os.makedirs(DIM_REDUCTION_PATH, exist_ok=True)
        variance_plot_path = os.path.join(
            DIM_REDUCTION_PATH, f'pca_variance_{source_str}.png')

        plot_pca_variance(
            variance_ratios,
            cumulative_variance,
            output_path=variance_plot_path
        )

        pca_reduced = PCA(n_components=n_components_optimal, random_state=0)
        X_pca = pca_reduced.fit_transform(X)

        pca_columns = [f'PC{i+1}' for i in range(n_components_optimal)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        df_dim_reduced = pd.concat([df_pca, y], axis=1)

        logger_pipe.info(
            f"PCA applied. Reduced to dimension number: {df_dim_reduced.shape}")

        for i in range(n_components_optimal):
            contribution_plot_path = os.path.join(
                DIM_REDUCTION_PATH, f'pca_comp_features_{source_str}_dim{i + 1}.png')

            plot_pca_feature_loadings(
                pca=pca_reduced,
                feature_names=feature_names,
                component_index=i,
                output_path=contribution_plot_path,
                n_features=n_features_plot
            )

        if n_components_optimal >= 2:
            scatter_plot_path = os.path.join(
                DIM_REDUCTION_PATH, f'pca_scatter_2d_{source_str}.png')

            plot_pca_scatter(
                df_pca=df_dim_reduced,
                target_col=target_col,
                output_path=scatter_plot_path
            )

        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        output_filename = f'data_dim_reduced_{source_str}.csv'
        output_path = os.path.join(PROCESSED_DATA_PATH, output_filename)
        df_dim_reduced.to_csv(output_path, index=False)
        logger_pipe.info(f"Saved dimensionally reduced data to {output_path}")

    except Exception as e:
        logger_pipe.error(
            f"Error during dimensionality reduction pipeline: {e}")
        return None

    logger_pipe.info(
        f"Dimensionality reduction pipeline finished.")
    return df_dim_reduced
