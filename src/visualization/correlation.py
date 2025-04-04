import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from loggers import logger_visualizations as logger


def plot_correlation_matrix(df, output_dir='plots/01_basic_eda', filename=None):
    if df is None or df.empty:
        logger.error("No data to plot correlation matrix.")
        return

    logger.info(f"Plotting correlation matrix for all features")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f'uci_correlation_matrix.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Correlation matrix saved to {output_path}")
    plt.close()


def plot_correlation_coefficients(df, output_dir='plots/01_basic_eda', filename=None):
    if df is None or df.empty:
        logger.error("No data to plot feature correlation coefficients.")
        return

    logger.info(f"Plotting feature correlation coefficients")
    plt.figure(figsize=(12, 8))
    df.corr()['target'].drop('target').sort_values().plot.barh()
    plt.title('Feature Correlation with target')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f'uci_feature_correlation_coefficients.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(
        f"Feature correlation coefficients plot saved to {output_path}")
    plt.close()


def plot_data_correlation(df, output_dir='plots/01_basic_eda'):
    if df is None or df.empty:
        logger.error("No data to plot feature correlation.")
        return

    logger.info("Plotting data feature correlation...")
    plot_correlation_matrix(df, output_dir)
    plot_correlation_coefficients(df, output_dir)
