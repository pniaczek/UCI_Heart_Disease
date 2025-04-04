import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from loggers import logger_visualizations as logger


def plot_pca_variance(variance_ratios, cumulative_variance, output_path):
    component_numbers = np.arange(1, len(variance_ratios) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(component_numbers, variance_ratios, c='r',
             marker='o', label='explained variance')
    plt.plot(component_numbers, cumulative_variance, c='b',
             marker='o', label='cumulative variance')
    plt.xlabel('Principal component k number')
    plt.ylabel('Explained variance ratio')
    plt.title('Variances vs PC number')
    plt.xticks(component_numbers)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        logger.info(f"Saved PCA variance plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save PCA variance plot: {e}")

    plt.close()


def plot_pca_feature_loadings(pca, feature_names, component_index, output_path, n_features=10):
    try:
        component = pca.components_[component_index]
        feature_importance = pd.Series(
            component, index=feature_names).sort_values(ascending=False)

        top_features = feature_importance.head(n_features)
        bottom_features = feature_importance.tail(n_features).sort_values(
            ascending=True)

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        sns.barplot(x=top_features.values,
                    y=top_features.index, hue=top_features.index, palette="viridis", legend=False)
        plt.title(
            f'Top {n_features} Feature Loadings for PC {component_index + 1}')
        plt.xlabel('Loading')
        plt.ylabel('')

        plt.subplot(1, 2, 2)
        sns.barplot(x=bottom_features.values,
                    y=bottom_features.index, hue=bottom_features.index, palette="viridis", legend=False)
        plt.title(
            f'Last {n_features} Feature Loadings for PC {component_index + 1}')
        plt.xlabel('Loading')
        plt.ylabel('')

        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(
            f"Saved PCA feature loadings plot for PC {component_index + 1} to {output_path}")

    except Exception as e:
        logger.error(
            f"Failed to save PCA feature loadings plot for PC {component_index + 1}: {e}")

    plt.close()


def plot_pca_scatter(df_pca, target_col, output_path):
    try:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2',
                        hue=target_col, palette='viridis', alpha=0.7)
        plt.title('PCA on First Two Principal Components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title=target_col)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Saved PCA 2D scatter plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save PCA 2D scatter plot: {e}")

    plt.close()
