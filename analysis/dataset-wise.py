import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

dataset = 'dbpedia'
# Define models and benchmarks
models = [
    'Qwen2.5',
    'LLAMA3.1',
    'Mistral',
    'Gemma2',
    'Gpt-4o mini',
    'AGG Gpt-4o',
    'AGG ConsUp',
    'AGG ConsDown',
]

benchmarks = ['DKA', 'GIV-Z', 'GIV-F', 'RAG']

# Seed for reproducibility
np.random.seed(42)

# Create data array with the right shape: (benchmarks, models)
data = np.zeros((len(benchmarks), len(models)))


# Fill with actual balanced accuracy data (multiplied by 100)
# DKA data
data[0, 0] = 63  # Qwen2.5
data[0, 1] = 58  # LLAMA3.1
data[0, 2] = 63  # Mistral
data[0, 3] = 64  # Gemma2

data[0, 4] = 61  # OpenAI (GPT-4o mini from first table)
data[0, 5] = 66  # AGG-OpenAI (GPT-4o mini from second table)
data[0, 6] = 64  # AGG-ConsUP (Cons_M ↑)
data[0, 7] = 66  # AGG-ConsDown (Cons_M ↓)

# GIV-Z data
data[1, 0] = 63  # Qwen2.5
data[1, 1] = 60  # LLAMA3.1
data[1, 2] = 55  # Mistral
data[1, 3] = 65  # Gemma2

data[1, 4] = 61  # OpenAI (GPT-4o mini from first table)
data[1, 5] = 66  # AGG-OpenAI (GPT-4o mini from second table)
data[1, 6] = 67  # AGG-ConsUP (Cons_M ↑)
data[1, 7] = 66  # AGG-ConsDown (Cons_M ↓)

# GIV-F data
data[2, 0] = 65  # Qwen2.5
data[2, 1] = 62  # LLAMA3.1
data[2, 2] = 54  # Mistral
data[2, 3] = 63  # Gemma2

data[2, 4] = 59  # OpenAI (GPT-4o mini from first table)
data[2, 5] = 66  # AGG-OpenAI (GPT-4o mini from second table)
data[2, 6] = 67  # AGG-ConsUP (Cons_M ↑)
data[2, 7] = 66  # AGG-ConsDown (Cons_M ↓)

# RAG data
data[3, 0] = 67  # Qwen2.5
data[3, 1] = 62  # LLAMA3.1
data[3, 2] = 66  # Mistral
data[3, 3] = 67  # Gemma2

data[3, 4] = 67  # OpenAI (GPT-4o mini from first table)
data[3, 5] = 67  # AGG-OpenAI (GPT-4o mini from second table)
data[3, 6] = 67  # AGG-ConsUP (Cons_M ↑)
data[3, 7] = 68  # AGG-ConsDown (Cons_M ↓)

# Define colors for each model
colors = [
    '#E6E6E6',  # Qwen2.5
    '#B3C6FF',  # LLAMA3.1
    '#FFD6B3',  # Mistral
    '#C6E6C6',  # Gemma2
    '#FFB3B3',  # OpenAI
    '#6699CC',  # AGG-OpenAI
    '#5DAE5D',  # AGG-ConsUP
    '#CC7A7A'   # AGG-ConsDown
]

# Setup plot
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Bar positioning
x = np.arange(len(benchmarks))
width = 0.1
offsets = np.linspace(-3.5, 3.5, len(models))

# Rounded bar function
def create_rounded_bar(ax, x, height, width, color, hatch=None, edgecolor=None, linewidth=0):
    radius = width * 0.25
    patch = FancyBboxPatch(
        (x - width/2, 0), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, alpha=0.9, hatch=hatch,
        edgecolor=edgecolor if edgecolor else color,
        linewidth=linewidth
    )
    ax.add_patch(patch)
    return patch

# Draw all bars with labels
for i, model in enumerate(models):
    is_agg = model.startswith('AGG')
    for j in range(len(benchmarks)):
        xpos = x[j] + offsets[i] * width
        height = data[j, i]

        create_rounded_bar(
            ax, xpos, height, width,
            color=colors[i],
            hatch='///' if is_agg else None,
            edgecolor='#555555' if is_agg else None,
            linewidth=1.0 if is_agg else 0
        )

        # Add label on top
        ax.text(
            xpos, height + 1, f'{int(height)}',
            ha='center', va='bottom',
            fontsize=8, fontweight='normal',
            color='#333333'
        )

# Axes and ticks
ax.set_ylabel('Balanced Accuracy (BAcc)', fontsize=10, color='#555555')
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=10, color='#555555')
ax.tick_params(axis='y', colors='#777777', labelsize=9)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#EEEEEE', zorder=0)
ax.set_axisbelow(True)

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Custom legend
legend_handles = []
for i, model in enumerate(models):
    is_agg = model.startswith('AGG')
    patch = mpatches.Patch(
        facecolor=colors[i],
        label=model,
        hatch='///' if is_agg else None,
        edgecolor='#555555' if is_agg else None,
        linewidth=1.0 if is_agg else 0
    )
    legend_handles.append(patch)

ax.legend(
    handles=legend_handles, loc='lower center',
    bbox_to_anchor=(0.5, 1.05), ncol=4,
    frameon=False, fontsize=9, handlelength=1.5
)

# X-axis padding for aesthetics
ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f'analysis/{dataset}_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()