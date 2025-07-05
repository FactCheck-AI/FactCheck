import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data preparation - organized by model, then dataset, then method
data = {
    'Gemma2:9b': {
        'FactBench': {'DKA': 0.75, 'GIV-Z': 0.74, 'GIV-F': 0.77, 'RAG': 0.90},
        'YAGO': {'DKA': 0.53, 'GIV-Z': 0.58, 'GIV-F': 0.52, 'RAG': 0.56},
        'DBpedia': {'DKA': 0.64, 'GIV-Z': 0.65, 'GIV-F': 0.63, 'RAG': 0.67}
    },
    'Qwen2.5:7b': {
        'FactBench': {'DKA': 0.67, 'GIV-Z': 0.65, 'GIV-F': 0.74, 'RAG': 0.87},
        'YAGO': {'DKA': 0.59, 'GIV-Z': 0.64, 'GIV-F': 0.64, 'RAG': 0.57},
        'DBpedia': {'DKA': 0.63, 'GIV-Z': 0.63, 'GIV-F': 0.65, 'RAG': 0.67}
    },
    'Mistral:7b': {
        'FactBench': {'DKA': 0.72, 'GIV-Z': 0.74, 'GIV-F': 0.77, 'RAG': 0.84},
        'YAGO': {'DKA': 0.44, 'GIV-Z': 0.53, 'GIV-F': 0.46, 'RAG': 0.51},
        'DBpedia': {'DKA': 0.63, 'GIV-Z': 0.55, 'GIV-F': 0.54, 'RAG': 0.66}
    },
    'Llama3.1:8b': {
        'FactBench': {'DKA': 0.74, 'GIV-Z': 0.65, 'GIV-F': 0.73, 'RAG': 0.82},
        'YAGO': {'DKA': 0.55, 'GIV-Z': 0.59, 'GIV-F': 0.58, 'RAG': 0.51},
        'DBpedia': {'DKA': 0.58, 'GIV-Z': 0.60, 'GIV-F': 0.62, 'RAG': 0.62}
    },
    'Gpt-4o-mini': {
        'FactBench': {'DKA': 0.66, 'GIV-Z': 0.65, 'GIV-F': 0.65, 'RAG': 0.90},
        'YAGO': {'DKA': 0.57, 'GIV-Z': 0.58, 'GIV-F': 0.63, 'RAG': 0.54},
        'DBpedia': {'DKA': 0.61, 'GIV-Z': 0.61, 'GIV-F': 0.59, 'RAG': 0.67}
    },
    'Cons-up': {
        'FactBench': {'DKA': 0.73, 'GIV-Z': 0.76, 'GIV-F': 0.80, 'RAG': 0.90},
        'YAGO': {'DKA': 0.53, 'GIV-Z': 0.64, 'GIV-F': 0.54, 'RAG': 0.53},
        'DBpedia': {'DKA': 0.64, 'GIV-Z': 0.67, 'GIV-F': 0.67, 'RAG': 0.67}
    },
    'Cons-down': {
        'FactBench': {'DKA': 0.73, 'GIV-Z': 0.71, 'GIV-F': 0.80, 'RAG': 0.90},
        'YAGO': {'DKA': 0.55, 'GIV-Z': 0.60, 'GIV-F': 0.55, 'RAG': 0.54},
        'DBpedia': {'DKA': 0.66, 'GIV-Z': 0.66, 'GIV-F': 0.66, 'RAG': 0.68}
    },
    'Agg-Gpt-4o-mini': {
        'FactBench': {'DKA': 0.74, 'GIV-Z': 0.71, 'GIV-F': 0.80, 'RAG': 0.90},
        'YAGO': {'DKA': 0.49, 'GIV-Z': 0.60, 'GIV-F': 0.55, 'RAG': 0.53},
        'DBpedia': {'DKA': 0.66, 'GIV-Z': 0.66, 'GIV-F': 0.66, 'RAG': 0.67}
    }
}

# Set up the figure with academic styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 5, figsize=(25, 10), subplot_kw=dict(projection='polar'))

# Methods (will be the radar chart axes)
methods = ['DKA', 'GIV-Z', 'GIV-F', 'RAG']
num_methods = len(methods)

# Calculate angles for each method on the radar chart
angles = np.linspace(0, 2 * np.pi, num_methods, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Colors and line styles for datasets
dataset_colors = {'FactBench': '#1f77b4', 'YAGO': '#ff7f0e', 'DBpedia': '#2ca02c'}
dataset_styles = {'FactBench': '-', 'YAGO': '--', 'DBpedia': '-.'}
datasets = ['FactBench', 'YAGO', 'DBpedia']

# Models and labels
individual_models = ['Gemma2:9b', 'Qwen2.5:7b', 'Mistral:7b', 'Llama3.1:8b', 'Gpt-4o-mini']
consensus_models = ['Cons-up', 'Cons-down', 'Agg-Gpt-4o-mini']
all_models = individual_models + consensus_models
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# Models and labels
individual_models = ['Gemma2:9b', 'Qwen2.5:7b', 'Mistral:7b', 'Llama3.1:8b', 'Gpt-4o-mini']
consensus_models = ['Cons-up', 'Cons-down', 'Agg-Gpt-4o-mini']
all_models = individual_models + consensus_models
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

for idx, model in enumerate(all_models):
    # Determine row and column
    row, col = (0, idx) if idx < 5 else (1, idx - 5 + 1)
    ax = axes[row, col]

    # Plot each dataset as a separate line
    for dataset in datasets:
        # Get values for this model and dataset
        values = [data[model][dataset][method] for method in methods]
        values += values[:1]  # Complete the circle

        # Plot the radar chart
        ax.plot(angles, values, 'o-', linewidth=1.5, label=dataset,
                color=dataset_colors[dataset], linestyle=dataset_styles[dataset])
        ax.fill(angles, values, alpha=0.15, color=dataset_colors[dataset])

        # # Add value labels on the plot
        # for angle, value, method in zip(angles[:-1], values[:-1], methods):
        #     ax.text(angle, value + 0.02, f'{value:.2f}',
        #             horizontalalignment='center', verticalalignment='center',
        #             fontsize=7, fontweight='bold', color=dataset_colors[dataset])

    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8,  1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)

    # Add title below the plot
    ax.text(
        0.5,
        -0.1,
        f'{labels[idx]}) {model}, balanced accuracy',
        transform=ax.transAxes,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=10
    )

# Remove unused subplots in the second row (first and last positions)
axes[1, 0].remove()
axes[1, 4].remove()

# Add legend
axes[0, 0].legend(loc='lower right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

# Adjust layout
plt.tight_layout()

plt.savefig('analysis/model_performance_radar_chart.png', dpi=300, bbox_inches='tight')

# Create a summary table
print("\nIndividual Models Performance Summary:")
print("=" * 80)
for model in individual_models:
    print(f"\n{model}:")
    model_df = pd.DataFrame(data[model]).round(3)
    print(model_df.to_string())

print("\n\nConsensus Models Performance Summary:")
print("=" * 80)
for model in consensus_models:
    print(f"\n{model}:")
    model_df = pd.DataFrame(data[model]).round(3)
    print(model_df.to_string())