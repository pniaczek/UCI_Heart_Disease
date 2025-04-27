import os
import pandas as pd
from src.data_processing.processor import Processor
from src.visualization.missing_values import plot_missing_values_bar_plot, plot_missing_values_matrix_plot, plot_missing_values_heatmap
from loggers import logger_main as logger


CONFIG_PATH = 'config/data_config.yaml'
PROCESSED_DATA_PATH = 'data/processed'
EDA_PATH = 'plots/01_basic_eda'


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
