"""
Visualization functions for RAG_concept_net results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_accuracy_boxplots_all_prompts(
    data_df,
    dataset_names,
    prompt_order,
    output_path=None,
    figsize=(14, 14),
    dpi=300,
    custom_labels=None
):
    """
    Similar to create_accuracy_boxplots but shows all prompt types without filtering.
    This ensures all prompts from prompt_order are displayed even if some have no data.
    
    Parameters:
    -----------
    custom_labels : list, optional
        Custom display names for x-axis labels. If None, uses prompt_order names.
        Must be same length as prompt_order.
    """
    # Set up the color palette for rounds
    color_palette = sns.color_palette("Accent")
    
    # Create figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(len(dataset_names), 1, figsize=figsize, sharex=True)
    
    # Label for each subplot
    subplot_labels = [f'({chr(65 + i)})' for i in range(len(dataset_names))]
    
    for idx, (ax, dataset, label) in enumerate(zip(axes, dataset_names, subplot_labels)):
        # Filter data for this dataset
        dataset_df = data_df[data_df['subject'] == dataset].copy()
        
        # Ensure prompt_type is categorical with specified order
        dataset_df['prompt_type'] = pd.Categorical(
            dataset_df['prompt_type'], 
            categories=prompt_order, 
            ordered=True
        )
        
        # Sort by prompt_type to maintain order
        dataset_df = dataset_df.sort_values('prompt_type')
        
        # Create box plot data for all prompts (even if empty)
        box_data = []
        for pt in prompt_order:
            if pt in dataset_df['prompt_type'].values:
                box_data.append(dataset_df[dataset_df['prompt_type'] == pt]['accuracy'].values)
            else:
                box_data.append([])  # Empty list for missing prompts
        
        # Create box plot (transparent light grey)
        box_parts = ax.boxplot(
            box_data,
            positions=range(len(prompt_order)),
            widths=0.8,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor='lightgrey', alpha=0.6, edgecolor='grey'),
            medianprops=dict(color='grey', linewidth=2.5),
            whiskerprops=dict(color='grey', linewidth=2.5),
            capprops=dict(color='grey', linewidth=2.5)
        )
        
        # Add individual data points with jitter
        x_positions = []
        y_values = []
        colors = []
        
        for i, prompt_type in enumerate(prompt_order):
            if prompt_type not in dataset_df['prompt_type'].values:
                continue
                
            prompt_data = dataset_df[dataset_df['prompt_type'] == prompt_type]
            
            # Add jitter to x positions
            n_points = len(prompt_data)
            jitter = np.random.uniform(-0.3, 0.3, n_points)
            x_pos = np.full(n_points, i) + jitter
            
            x_positions.extend(x_pos)
            y_values.extend(prompt_data['accuracy'].values)
            
            # Map round numbers to colors
            for round_num in prompt_data['round'].values:
                colors.append(color_palette[int(round_num) - 1])
        
        # Scatter plot for individual points
        ax.scatter(x_positions, y_values, c=colors, alpha=0.7, s=80, edgecolors='black', linewidth=0.4, zorder=3)
        
        # Formatting
        ax.set_ylabel('Accuracy', fontsize=15, fontweight='bold', labelpad=10)
        ax.set_title(f'{label} {dataset.replace("_", " ").title()}', fontsize=20, fontweight='bold', loc='left')
        ax.set_ylim(0.7, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Set x-axis labels only for the bottom plot
        if idx == len(dataset_names) - 1:  # Bottom plot (more flexible than idx == 2)
            ax.set_xticks(range(len(prompt_order)))
            # Use custom labels if provided, otherwise use prompt_order
            labels_to_display = custom_labels if custom_labels is not None else prompt_order
            ax.set_xticklabels(labels_to_display, rotation=45, ha='right', fontsize=17)
            ax.set_xlabel('Prompt Type', fontsize=15, fontweight='bold', labelpad=5)
        else:
            ax.set_xticks(range(len(prompt_order)))
            ax.set_xticklabels([])
    
    # Create custom legend for rounds
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_palette[i], 
                   markersize=8, label=f'Round {i+1}', markeredgecolor='black', markeredgewidth=0.5)
        for i in range(5)
    ]
    
    # Add legend to the top subplot
    axes[0].legend(handles=legend_elements, loc='upper right', ncol=5, frameon=True, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)  # Increase this value for more space
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    return fig, axes
