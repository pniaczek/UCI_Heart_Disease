import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)-8s] :: %(asctime)s :: %(name)-15s :: %(message)s'
)

logger_main = logging.getLogger("Main")
logger_processor = logging.getLogger("Processor")
logger_imputer = logging.getLogger("Imputer")
logger_transformer = logging.getLogger("Transformer")  # Added
logger_pipelines = logging.getLogger("Pipelines")  # Added
logger_visualizations = logging.getLogger("Visualizations")
