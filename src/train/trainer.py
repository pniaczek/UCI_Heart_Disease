import yaml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from loggers import logger_trainer as logger



class Trainer:
    def __init__(self, X, y, training_config_path):
        self.X = X
        self.y = y
        self.config = self._load_config(training_config_path)
        self._prepare_data()
        self.learning_curve = {
            'accuracy': [],
            'recall': [],
            'f1_score': []
        }

    def _load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def _prepare_data(self):
        logger.info("Splitting dataset into train, validation, and test sets...")
        train_ratio = self.config.get("train_ratio", 0.8)
        val_ratio = self.config.get("val_ratio", 0.1)
        test_ratio = self.config.get("test_ratio", 0.1)

        if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
            logger.warning("Train/val/test ratios do not sum to 1.0. Skipping data preparation.")
            self.X_train = self.X_val = self.X_test = None
            self.y_train = self.y_val = self.y_test = None
            return

        # test data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_ratio,
            stratify=self.y,
            random_state=42
        )

        # train and val data
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=42
        )

    def train(self, model_name: str):
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

        logger.info("SVM model training completed. Evaluating on validation set...")
        y_pred = model.predict(self.X_val)

        acc = accuracy_score(self.y_val, y_pred)
        rec = recall_score(self.y_val, y_pred, average='weighted', zero_division=0)
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
