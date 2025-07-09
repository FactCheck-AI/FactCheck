import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data preparation
data = {
    'BACC': {
        'DKA': {
            'gemma2:9b': 0.70,
            'qwen2.5:7b': 0.65,
            'llama3.1:8b': 0.66,
            'mistral:7b': 0.68,
            'gpt-4o-mini': 0.64,
            'agg-cons-up': 0.70,
            'agg-cons-down': 0.70,
            'agg-gpt-4o-mini': 0.70
        },
        'GIV-Z': {
            'gemma2:9b': 0.70,
            'qwen2.5:7b': 0.65,
            'llama3.1:8b': 0.62,
            'mistral:7b': 0.64,
            'gpt-4o-mini': 0.63,
            'agg-cons-up': 0.70,
            'agg-cons-down': 0.70,
            'agg-gpt-4o-mini': 0.69
        },
        'GIV-F': {
            'gemma2:9b': 0.71,
            'qwen2.5:7b': 0.68,
            'llama3.1:8b': 0.65,
            'mistral:7b': 0.65,
            'gpt-4o-mini': 0.61,
            'agg-cons-up': 0.72,
            'agg-cons-down': 0.73,
            'agg-gpt-4o-mini': 0.72
        },
        'RAG': {
            'gemma2:9b': 0.75,
            'qwen2.5:7b': 0.74,
            'llama3.1:8b': 0.69,
            'mistral:7b': 0.72,
            'gpt-4o-mini': 0.75,
            'agg-cons-up': 0.75,
            'agg-cons-down': 0.75,
            'agg-gpt-4o-mini': 0.75
        }
    },
    "F1": {
        'DKA': {
            'gemma2:9b': 0.66,
            'qwen2.5:7b': 0.50,
            'llama3.1:8b': 0.61,
            'mistral:7b': 0.60,
            'gpt-4o-mini': 0.48,
            'agg-cons-up': 0.64,
            'agg-cons-down': 0.62,
            'agg-gpt-4o-mini': 0.61
        },
        'GIV-Z': {
            'gemma2:9b': 0.65,
            'qwen2.5:7b': 0.51,
            'llama3.1:8b': 0.46,
            'mistral:7b': 0.64,
            'gpt-4o-mini': 0.45,
            'agg-cons-up': 0.61,
            'agg-cons-down': 0.59,
            'agg-gpt-4o-mini': 0.58
        },
        'GIV-F': {
            'gemma2:9b': 0.68,
            'qwen2.5:7b': 0.61,
            'llama3.1:8b': 0.57,
            'mistral:7b': 0.66,
            'gpt-4o-mini': 0.40,
            'agg-cons-up': 0.68,
            'agg-cons-down': 0.69,
            'agg-gpt-4o-mini': 0.66
        },
        'RAG': {
            'gemma2:9b': 0.68,
            'qwen2.5:7b': 0.69,
            'llama3.1:8b': 0.62,
            'mistral:7b': 0.68,
            'gpt-4o-mini': 0.66,
            'agg-cons-up': 0.69,
            'agg-cons-down': 0.70,
            'agg-gpt-4o-mini': 0.69
        }
    }
}



# Set up the figure with academic styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(projection='polar'))

# Models (will be the radar chart axes)
models = list(data['BACC']['DKA'].keys())
num_models = len(models)

# Calculate angles for each model on the radar chart
angles = np.linspace(0, 2 * np.pi, num_models, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Colors and styles for each method
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']
method_names = ['DKA', 'GIV-Z', 'GIV-F', 'RAG']

# Metric names for titles
metric_names = ['BACC', 'F1']
metric_titles = ['Balanced Accuracy', 'F1 Score']

# Create both charts
for metric_idx, metric in enumerate(metric_names):
    ax = axes[metric_idx]

    # Plot each method on the radar chart
    for idx, method in enumerate(method_names):
        # Get values for this method
        values = [data[metric][method][model] for model in models]
        values += values[:1]  # Complete the circle

        # Plot the radar chart
        ax.plot(angles, values,
                linestyle=line_styles[idx],
                marker=markers[idx],
                linewidth=3,
                markersize=8,
                label=method,
                color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models, fontsize=10, fontweight='bold')
    ax.set_ylim(0.4, 0.8)
    # set the yticks labels one in a row
    yticks = np.arange(0.4, 0.81, 0.05)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add title for each subplot
    ax.text(0.5, -0.08, f'{metric_titles[metric_idx]}',
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=13, fontweight='bold')


    # Add legend only to the second chart to avoid duplication
    if metric_idx == 0:
        ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(-0.25, 1.2), fontsize=12, frameon=True)

# Add overall title
# fig.suptitle('Model Performance Comparison: BACC vs F1 Scores',
#              fontsize=16, fontweight='bold', y=0.95)

# Adjust layout to accommodate legend
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save the figure
plt.savefig('analysis/bacc_f1_comparison.png', bbox_inches='tight', dpi=600)
plt.show()

# Create summary comparison tables for both metrics
for metric in metric_names:
    print(f"\n{metric} Summary Table:")
    print("=" * 60)
    df = pd.DataFrame(data[metric]).round(3)
    print(df.to_string())

    print(f"\n{metric} Summary Statistics:")
    print("=" * 40)
    for method in method_names:
        scores = list(data[metric][method].values())
        print(f"{method:6s}: Mean={np.mean(scores):.3f}, Std={np.std(scores):.3f}, "
              f"Max={np.max(scores):.3f}, Min={np.min(scores):.3f}")

    print(f"\n{metric} Best Performing Models:")
    print("=" * 40)
    for method in method_names:
        best_model = max(data[metric][method], key=data[metric][method].get)
        best_score = data[metric][method][best_model]
        print(f"{method:6s}: {best_model} ({best_score:.3f})")
    print("\n" + "="*60)