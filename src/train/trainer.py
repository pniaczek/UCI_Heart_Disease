import yaml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from loggers import logger_trainer as logger


class Trainer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, training_config_path):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.config = self._load_config(training_config_path)
        # Data is already prepared and split
        self.learning_curve = {
            'accuracy': [],
            'recall': [],
            'f1_score': []
        }

    def _load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    # _prepare_data method is removed as splitting is done externally

    def train(self, model_name: str):
        # Check if data was loaded correctly during init
        if self.X_train is None or self.y_train is None or self.X_val is None or self.y_val is None:
            logger.error(
                "Training data is not available. Check the preprocessing steps.")
            return None, None

        if model_name not in self.config.get('models', {}):
            logger.warning(f"No training config for model '{model_name}'")
            return None, None

        train_fn = getattr(self, f'_train_{model_name}', None)
        if train_fn is None:
            logger.warning(f"No training method for model '{model_name}'")
            return None, None

        model_config = self.config['models'][model_name]
        return train_fn(model_config)

    def _train_svm(self, config):
        logger.info(f"Initializing SVM model with config: {config}")
        model = SVC(**config)

        logger.info("Training SVM model...")
        model.fit(self.X_train, self.y_train)

        logger.info("Evaluating on training set...")
        y_train_pred = model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, y_train_pred)
        logger.info(f"Training Accuracy: {train_acc:.4f}")

        logger.info(
            "SVM model training completed. Evaluating on validation set...")
        y_pred = model.predict(self.X_val)

        acc = accuracy_score(self.y_val, y_pred)
        rec = recall_score(self.y_val, y_pred,
                           average='weighted', zero_division=0)
        f1 = f1_score(self.y_val, y_pred, average='weighted', zero_division=0)

        logger.info(f"Validation Accuracy: {acc:.4f}")
        logger.info(f"Validation Recall: {rec:.4f}")
        logger.info(f"Validation F1 Score: {f1:.4f}")

        self.learning_curve['accuracy'].append(acc)
        self.learning_curve['recall'].append(rec)
        self.learning_curve['f1_score'].append(f1)

        return model, {
            'accuracy': acc,
            'recall': rec,
            'f1_score': f1
        }
