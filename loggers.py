import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)-8s] :: %(asctime)s :: %(name)-15s :: %(message)s'
)

logger_main = logging.getLogger("Main")
logger_processor = logging.getLogger("Processor")
logger_imputer = logging.getLogger("Imputer")
logger_transformer = logging.getLogger("Transformer")
logger_pipelines = logging.getLogger("Pipelines")
logger_visualizations = logging.getLogger("Visualizations")
logger_trainer = logging.getLogger("Training")
