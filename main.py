from src.pipelines.pipelines import (
    missing_data_visualization_pipeline,
    processing_pipeline,
    transformation_pipeline,
    dim_reduction_pipeline
)
from loggers import logger_main as logger


def main():
    logger.info("Starting analysis...")

    missing_data_visualization_pipeline()
    df_processed = processing_pipeline()

    if df_processed is None:
        logger.error("Processing pipeline failed.")
        return

    df_transformed = transformation_pipeline(df_processed)
    if df_transformed is None:
        logger.error("Transformation pipeline failed.")
        return

    df_dim_reduced = dim_reduction_pipeline(
        df_transformed, variance_threshold=0.9)
    if df_dim_reduced is None:
        logger.error("Dimensionality reduction pipeline failed.")
        return

    logger.info("Analysis finished.")


if __name__ == "__main__":
    main()
