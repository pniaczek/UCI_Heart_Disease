import pandas as pd
import yaml
import optuna
import importlib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path='config/model_config.yaml'):
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_model_class(self, model_string):
        module_name, class_name = model_string.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def train_evaluate_model(self, model_name, X_train, y_train, X_test, y_test, best_params):
        model_config = self.config[model_name]
        model_class = self.get_model_class(model_config['model'])
        model = model_class(**model_config.get('params', {}), **best_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X_train.columns, model.coef_[0]))

        return metrics, cm, feature_importance, model

    def objective(self, trial, model_name, X, y):
        model_config = self.config[model_name]
        model_class = self.get_model_class(model_config['model'])
        tuning_params = model_config.get('tuning', {})

        params = {}
        for param_name, param_info in tuning_params.items():
            if param_info['type'] == 'float':
                if param_info.get('log'):
                    params[param_name] = trial.suggest_loguniform(
                        param_name, param_info['low'], param_info['high'])
                else:
                    params[param_name] = trial.suggest_uniform(
                        param_name, param_info['low'], param_info['high'])
            elif param_info['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_info['low'], param_info['high'], step=param_info.get('step', 1))
            elif param_info['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_info['choices'])

        model = model_class(**model_config.get('params', {}), **params)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        cv_scores = []
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            cv_scores.append(accuracy_score(y_val, y_pred))

        return np.mean(cv_scores)

    def tune_models(self, X_train, y_train):
        tuned_models = {}
        for model_name in self.config.keys():
            logger.info(f"Tuning model: {model_name}")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(
                trial, model_name, X_train, y_train), n_trials=50)
            tuned_models[model_name] = study.best_params
            logger.info(
                f"Best hyperparameters for {model_name}: {study.best_params}")
        return tuned_models

    def run_training_and_evaluation(self, X_train, y_train, X_test, y_test):
        tuned_params = self.tune_models(X_train, y_train)
        evaluation_results = {}
        confusion_matrices = {}
        feature_importances = {}
        trained_models = {}

        for model_name, best_params in tuned_params.items():
            logger.info(
                f"Training and evaluating model: {model_name} with best params: {best_params}")
            metrics, cm, feature_importance, model = self.train_evaluate_model(
                model_name, X_train, y_train, X_test, y_test, best_params)
            evaluation_results[model_name] = {
                'metrics': metrics,
                'hyperparameters': best_params
            }
            confusion_matrices[model_name] = cm
            feature_importances[model_name] = feature_importance
            trained_models[model_name] = model

        return evaluation_results, confusion_matrices, feature_importances, trained_models
