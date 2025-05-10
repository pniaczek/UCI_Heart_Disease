import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from loggers import logger_main as logger


def plot_metric_comparison(results, output_dir, source_str=''):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    splits = ['train', 'val', 'test']
    model_names = list(results.keys())

    for split in splits:
        plt.figure(figsize=(12, 8))
        plot_data = {metric: [] for metric in metrics}
        hyperparams_info = []

        for model_name in model_names:
            hyperparams_info.append(
                f"{model_name.upper()} Params: {results[model_name]['best_params']}")
            for metric in metrics:
                key = f'{split}_{metric}'
                plot_data[metric].append(
                    results[model_name]['metrics'].get(key, np.nan))

        df_plot = pd.DataFrame(plot_data, index=model_names)

        df_plot.plot(kind='bar', rot=0, figsize=(14, 7))
        plt.title(f'Model Comparison - {split.capitalize()} Set Metrics\n' + '\n'.join(
            hyperparams_info), fontsize=10)
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.ylim(0, 1.05)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        filename = os.path.join(
            output_dir, f'{source_str}comparison_{split}_metrics.png')
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved {split} metrics comparison plot to {filename}")


def plot_confusion_matrix(model, X_test, y_test, model_name, class_names, output_dir, source_str=''):
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name.upper()} (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    filename = os.path.join(
        output_dir, f'{source_str}confusion_matrix_{model_name}.png')
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved confusion matrix for {model_name} to {filename}")


def plot_feature_importance(model, feature_names, model_name, output_dir, top_n=5, source_str=''):
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(model, 'feature_importances_'):
        logger.warning(
            f"Model {model_name} does not support feature importances.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances - {model_name.upper()}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')

    for index, value in enumerate(top_importances):
        plt.text(value, index, f'{value:.3f}')

    plt.tight_layout()
    filename = os.path.join(
        output_dir, f'{source_str}feature_importance_{model_name}.png')
    plt.savefig(filename)
    plt.close()
    logger.info(
        f"Saved feature importance plot for {model_name} to {filename}")
