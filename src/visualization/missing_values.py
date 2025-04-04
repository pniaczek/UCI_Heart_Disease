import matplotlib.pyplot as plt
import seaborn as sns
import os
import missingno as msno
from loggers import logger_visualizations as logger


def plot_missing_values_bar_plot(df, filename, output_dir='plots/01_basic_eda'):
    plt.figure(figsize=(10, 8))
    ax = msno.bar(df)
    plt.title("Missing Value Bar Plot per Feature", fontsize=26, pad=20)
    ax.set_ylabel("Proportion and Number of Non-Missing Values",
                  fontsize=22, labelpad=15)
    plt.xticks(rotation=45, ha="center")
    ax.grid(axis="y", linestyle="--", alpha=0.9, color="gray", zorder=0)
    ax.axhline(y=0, color="gray", alpha=0.9, zorder=0)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Missing values bar plot saved to {output_path}")
    plt.close()


def plot_missing_values_matrix_plot(df, filename, output_dir='plots/01_basic_eda'):
    plt.figure(figsize=(10, 8))
    msno.matrix(df)
    plt.title("Missing Data Matrix Plot", fontsize=20)
    plt.ylabel("Samples", fontsize=18)
    plt.xlabel("Features", fontsize=18)
    plt.xticks(rotation=45, ha="center")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Missing values matrix plot saved to {output_path}")
    plt.close()


def plot_missing_values_heatmap(df, filename, output_dir='plots/01_basic_eda'):
    if df is None or df.empty:
        logger.error("No data to plot missing values.")
        return

    logger.info("Plotting missing values heatmap...")
    plt.figure(figsize=(10, 8))
    msno.heatmap(df, cmap="coolwarm")
    plt.title('Missing Values Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Data Points')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Missing values heatmap saved to {output_path}")
    plt.close()
