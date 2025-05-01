import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
from IPython.display import display, HTML
import ipywidgets as widgets
from em_interp.lora_analysis.get_lora_scalars import get_scalar_stats

def create_lora_scalar_visualization(
        scalar_set_path: str, output_path: str = None, 
        context_window: int = 10, top_k: int = 10, 
        filter_system: bool = True
    ):
    """
    Create a JSON visualization of top/bottom tokens by scalar value with context.
    
    Args:
        scalar_set_path: Path to the saved scalar set
        output_path: Path to save the JSON output
        context_window: Number of tokens to include on each side
        top_k: Number of top/bottom tokens to include
    """
    # Load scalar set
    data = torch.load(scalar_set_path)
    scalar_set = data["scalar_set"]
    tokenized_prompts = data["tokenized_prompts"]
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            "prompt": prompt, 
            "token": token_info[0], 
            "token_idx": token_idx,
            "layer": layer, 
            "scalar": scalar
        }
        for prompt, token_dict in scalar_set.items()
        for token_idx, token_info in token_dict.items()
        for layer, scalar in token_info[1].items()
    ])

    # filter out tokens related to the system prompt
    if filter_system:
        token_filter_str = "system You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>Ċ<|im_start|>userĊassistant"
        df = df[~df["token"].isin([t for t in df["token"] if t.replace("Ġ", "") in token_filter_str])]
    
    # Create visualization data
    visualization = {}
    
    for layer in df["layer"].unique():
        layer_df = df[df["layer"] == layer]
        
        # Get distribution data
        scalar_values = layer_df["scalar"].values
        
        # Create distribution plot
        plt.figure(figsize=(8, 4))
        sns.histplot(scalar_values, kde=True)
        plt.title(f"Distribution of Scalar Values for {layer}")
        plt.xlabel("Scalar Value")
        plt.ylabel("Frequency")
        
        # Save plot to file
        plot_dir = Path(output_path).parent if output_path else Path("lora_visualizations")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{layer.replace('.', '_')}_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Get top and bottom tokens
        top_tokens = []
        for _, row in layer_df.sort_values(by="scalar", ascending=False).head(top_k).iterrows():
            context = get_token_context_with_highlight(
                tokenized_prompts[row["prompt"]], 
                row["token_idx"], 
                context_window,
                "#7eb7d6",  # Highlight color for top tokens
                round(float(row["scalar"]), 3)
            )
            top_tokens.append({
                "token": row["token"],
                "scalar": round(float(row["scalar"]), 3),
                "context": context
            })
        
        bottom_tokens = []
        for _, row in layer_df.sort_values(by="scalar", ascending=True).head(top_k).iterrows():
            context = get_token_context_with_highlight(
                tokenized_prompts[row["prompt"]], 
                row["token_idx"], 
                context_window,
                "#e37d64",  # Highlight color for bottom tokens
                round(float(row["scalar"]), 3)
            )
            bottom_tokens.append({
                "token": row["token"],
                "scalar": round(float(row["scalar"]), 3),
                "context": context
            })
        
        # Add layer data to visualization
        visualization[layer] = {
            "distribution_stats": {
                "mean": float(np.mean(scalar_values)),
                "std": float(np.std(scalar_values)),
                "min": float(np.min(scalar_values)),
                "max": float(np.max(scalar_values)),
                "plot_path": str(plot_path)
            },
            "top_tokens": top_tokens,
            "bottom_tokens": bottom_tokens
        }
    
    # Save to JSON
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(visualization, f, indent=2)
    
    return visualization

def get_token_context_with_highlight(
    tokenized_prompt: List[str], 
    token_idx: int, 
    context_window: int = 10,
    highlight_color: str = "blue",
    scalar_value: float = None
) -> List[Dict[str, Any]]:
    """
    Get context around a token with highlighting.
    
    Args:
        tokenized_prompt: List of tokens in the prompt
        token_idx: Index of the token to highlight
        context_window: Number of tokens to include on each side
        highlight_color: Color to highlight the target token
        scalar_value: Scalar value to display with the token
        
    Returns:
        List of token objects with text and formatting
    """
    start_idx = max(0, token_idx - context_window)
    end_idx = min(len(tokenized_prompt), token_idx + context_window + 1)
    
    context = []
    for i in range(start_idx, end_idx):
        token_obj = {"text": tokenized_prompt[i]}
        
        if i == token_idx:
            # Add highlighting to the target token (but don't add scalar value in the context)
            token_obj["highlight"] = highlight_color
            token_obj["opacity"] = 0.3  # Light highlighting to keep text readable
        
        context.append(token_obj)
    
    return context

def display_scalar_analysis_interactive(scalar_set_paths, display_names=None, tokens_to_display=None):
    """
    Display an interactive visualization of the scalar set analysis in the notebook.
    
    Args:
        scalar_set_paths: Dictionary mapping display names to paths, or list of paths
        display_names: Optional dictionary mapping file paths to display names (if scalar_set_paths is a list)
        tokens_to_display: Optional dictionary mapping display names to lists of token indices to show
    """
    # Handle different input formats
    if isinstance(scalar_set_paths, dict):
        # If dictionary provided, use it directly
        display_dict = scalar_set_paths
    else:
        # If list provided, create dictionary with optional display names
        if isinstance(scalar_set_paths, str):
            scalar_set_paths = [scalar_set_paths]
            
        if display_names is None:
            # Use filenames as display names
            display_dict = {Path(path).stem: path for path in scalar_set_paths}
        else:
            # Use provided display names
            display_dict = {display_names.get(path, Path(path).stem): path 
                           for path in scalar_set_paths}
    
    # Limit to 4 files maximum
    if len(display_dict) > 4:
        display_dict = dict(list(display_dict.items())[:4])
    
    # Load the scalar sets
    data_sets = []
    visualizations = []
    
    # Define colors for different files
    file_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (display_name, path) in enumerate(display_dict.items()):
        # Load data
        data = torch.load(path)
        data_sets.append({
            "scalar_set": data["scalar_set"],
            "tokenized_prompts": data["tokenized_prompts"],
            "path": path,
            "display_name": display_name,
            "color": file_colors[i % len(file_colors)]  # Assign color
        })
        
        # Create visualization data
        vis = create_lora_scalar_visualization(
            path, 
            output_path=None,  # Don't save to file
            context_window=10,
            top_k=10
        )
        visualizations.append(vis)
    
    # Find common layers across all files
    common_layers = set(visualizations[0].keys())
    for vis in visualizations[1:]:
        common_layers = common_layers.intersection(set(vis.keys()))
    
    # Clean up layer names by removing common prefix
    def clean_layer_name(name):
        # Remove common prefixes like 'base_model.model.model.'
        if name.startswith('base_model.model.model.'):
            return name[len('base_model.model.model.'):]
        return name
    
    # Create layer selector dropdown with cleaned names
    layers = list(common_layers)
    layer_dropdown = widgets.Dropdown(
        options=[(clean_layer_name(layer), layer) for layer in layers],
        description='Layer:',
        disabled=False,
        layout=widgets.Layout(width='400px')
    )
    
    # Create file selector checkboxes (only for histogram)
    file_checkboxes = []
    for i, data in enumerate(data_sets):
        checkbox = widgets.Checkbox(
            value=True,
            description=data["display_name"],
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='200px')
        )
        file_checkboxes.append(checkbox)
    
    # Create file selector widget
    file_selector = widgets.HBox(file_checkboxes)
    
    # Function to display token contexts
    def display_token_contexts(tokens_data, token_indices=None):
        # Filter tokens if indices are provided
        if token_indices is not None:
            filtered_tokens = [token for i, token in enumerate(tokens_data) if i in token_indices]
        else:
            filtered_tokens = tokens_data
            
        html_output = "<div style='max-width: 800px;'>"
        for token_data in filtered_tokens:
            # More compact display with smaller margins
            html_output += f"<div style='margin-bottom: 8px; padding: 5px; border: 1px solid #ddd; border-radius: 3px;'>"
            
            # Show only the scalar value in bold at the beginning, not the token
            html_output += f"<span style='font-weight: bold;'>{token_data['scalar']}</span> "
            
            # Context display
            for token_obj in token_data['context']:
                text = token_obj['text'].replace('Ġ', ' ')
                if 'highlight' in token_obj:
                    color = token_obj['highlight']
                    # Make highlighted text bold black with stronger background
                    html_output += f"<span style='background-color: {color}; color: black; font-weight: bold;'>{text}</span>"
                else:
                    html_output += text
            
            html_output += "</div>"
        
        html_output += "</div>"
        return HTML(html_output)
    
    # Create distribution plot widget and stats display
    plot_output = widgets.Output()
    stats_output = widgets.Output()  # New widget for statistics
    
    # Create output widgets for token displays
    file_token_outputs = []
    for _ in data_sets:
        file_token_outputs.append({
            "top": widgets.Output(),
            "bottom": widgets.Output()
        })
    
    # Create main tabs for top/bottom tokens
    main_tabs = widgets.Tab()
    main_tabs.children = [widgets.Tab(), widgets.Tab()]  # Tabs for top and bottom tokens
    main_tabs.set_title(0, 'Top Tokens')
    main_tabs.set_title(1, 'Bottom Tokens')
    
    # Create file tabs for each category
    top_file_tabs = main_tabs.children[0]
    bottom_file_tabs = main_tabs.children[1]
    
    # Set up file tabs
    top_file_tabs.children = [widgets.Output() for _ in data_sets]
    bottom_file_tabs.children = [widgets.Output() for _ in data_sets]
    
    # Set tab titles
    for i, data in enumerate(data_sets):
        top_file_tabs.set_title(i, data["display_name"])
        bottom_file_tabs.set_title(i, data["display_name"])
    
    # Function to update display when layer or file selection changes
    def update_display(*args):
        layer = layer_dropdown.value
        selected_files = [i for i, cb in enumerate(file_checkboxes) if cb.value]
        
        # Update distribution plot
        with plot_output:
            plot_output.clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            
            # Plot distributions for selected files
            for i in selected_files:
                data = data_sets[i]
                vis = visualizations[i]
                layer_data = vis[layer]
                
                # Create DataFrame for this file
                df = pd.DataFrame([
                    {
                        "prompt": prompt, 
                        "token": token_info[0], 
                        "token_idx": token_idx,
                        "layer": layer_name, 
                        "scalar": scalar
                    }
                    for prompt, token_dict in data["scalar_set"].items()
                    for token_idx, token_info in token_dict.items()
                    for layer_name, scalar in token_info[1].items()
                ])
                
                layer_df = df[df["layer"] == layer]
                scalar_values = layer_df["scalar"].values
                
                # Plot with file-specific color and low opacity
                sns.histplot(scalar_values, kde=True, color=data["color"], 
                             alpha=0.3, label=data["display_name"], element="step",
                             stat="density")
            
            # Use cleaned layer name for title
            clean_title = clean_layer_name(layer)
            plt.title(f"Distribution of Scalar Values for {clean_title}")
            plt.xlabel("Scalar Value")
            plt.ylabel("Probability Density")
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        # Update statistics display
        with stats_output:
            stats_output.clear_output(wait=True)
            
            # Create HTML table for statistics
            html = "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
            html += "<tr><th style='text-align:left; padding:5px; border-bottom:1px solid #ddd;'>File</th>"
            html += "<th style='text-align:right; padding:5px; border-bottom:1px solid #ddd;'>Expected Abs Value</th>"
            html += "<th style='text-align:right; padding:5px; border-bottom:1px solid #ddd;'>Mean</th>"
            html += "<th style='text-align:right; padding:5px; border-bottom:1px solid #ddd;'>Std Dev</th></tr>"
            
            for i in selected_files:
                data = data_sets[i]
                vis = visualizations[i]
                layer_data = vis[layer]
                stats = layer_data["distribution_stats"]
                
                # Calculate expected absolute value
                df = pd.DataFrame([
                    {
                        "scalar": scalar
                    }
                    for prompt, token_dict in data["scalar_set"].items()
                    for token_idx, token_info in token_dict.items()
                    for layer_name, scalar in token_info[1].items()
                    if layer_name == layer
                ])
                
                expected_abs = df["scalar"].abs().mean()
                
                # Add row with color-coded file name
                html += f"<tr>"
                html += f"<td style='text-align:left; padding:5px; color:{data['color']}; font-weight:bold;'>{data['display_name']}</td>"
                html += f"<td style='text-align:right; padding:5px;'>{expected_abs:.4f}</td>"
                html += f"<td style='text-align:right; padding:5px;'>{stats['mean']:.4f}</td>"
                html += f"<td style='text-align:right; padding:5px;'>{stats['std']:.4f}</td>"
                html += f"</tr>"
            
            html += "</table>"
            display(HTML(html))
        
        # Update token displays for ALL files (not just selected ones)
        for i, data in enumerate(data_sets):
            vis = visualizations[i]
            layer_data = vis[layer]
            display_name = data["display_name"]
            
            # Get token indices to display for this file
            top_indices = None
            bottom_indices = None
            if tokens_to_display and display_name in tokens_to_display:
                file_tokens = tokens_to_display[display_name]
                if "top" in file_tokens:
                    top_indices = file_tokens["top"]
                if "bottom" in file_tokens:
                    bottom_indices = file_tokens["bottom"]
            
            # Update top tokens tab
            with top_file_tabs.children[i]:
                top_file_tabs.children[i].clear_output(wait=True)
                display(display_token_contexts(layer_data["top_tokens"], top_indices))
            
            # Update bottom tokens tab
            with bottom_file_tabs.children[i]:
                bottom_file_tabs.children[i].clear_output(wait=True)
                display(display_token_contexts(layer_data["bottom_tokens"], bottom_indices))
    
    # Set up the layout
    layer_dropdown.observe(update_display, names='value')
    for checkbox in file_checkboxes:
        checkbox.observe(update_display, names='value')
    
    # Initialize with first layer
    if layers:
        update_display()
    
    # Display widgets
    display(widgets.VBox([
        widgets.HTML("<h2>LoRA Scalar Analysis</h2>"),
        widgets.HTML("<h3>Files to Display in Histogram:</h3>"),
        file_selector,
        widgets.HTML("<h3>Select Layer:</h3>"),
        layer_dropdown,
        plot_output,
        stats_output,  # Add the statistics output
        widgets.HTML("<h3>Token Analysis</h3>"),
        main_tabs
    ]))

def compare_scalar_magnitudes(scalar_set_paths, display_names=None):
    """
    Compare the expected absolute values of scalars across different layers for multiple datasets.
    
    Args:
        scalar_set_paths: Dictionary mapping display names to paths, or list of paths
        display_names: Optional dictionary mapping file paths to display names (if scalar_set_paths is a list)
    
    Returns:
        Two plots: 
        1. Expected absolute values across layers for each dataset
        2. Percentage differences from the first dataset
    """
    # Handle different input formats
    if isinstance(scalar_set_paths, dict):
        # If dictionary provided, use it directly
        display_dict = scalar_set_paths
    else:
        # If list provided, create dictionary with optional display names
        if isinstance(scalar_set_paths, str):
            scalar_set_paths = [scalar_set_paths]
            
        if display_names is None:
            # Use filenames as display names
            display_dict = {Path(path).stem: path for path in scalar_set_paths}
        else:
            # Use provided display names
            display_dict = {display_names.get(path, Path(path).stem): path 
                           for path in scalar_set_paths}
    
    # Define colors for different files
    file_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Load data and calculate statistics
    data_sets = []
    all_layers = set()
    
    for i, (display_name, path) in enumerate(display_dict.items()):
        # Load data
        data = torch.load(path)
        
        # Create layer-wise statistics
        layer_stats = {}
        
        # Group scalars by layer
        for prompt, token_dict in data["scalar_set"].items():
            for token_idx, token_info in token_dict.items():
                for layer_name, scalar in token_info[1].items():
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = []
                    layer_stats[layer_name].append(scalar)
        
        # Calculate expected absolute value for each layer
        layer_abs_values = {}
        for layer_name, scalars in layer_stats.items():
            layer_abs_values[layer_name] = np.mean(np.abs(scalars))
            all_layers.add(layer_name)
        
        data_sets.append({
            "display_name": display_name,
            "color": file_colors[i % len(file_colors)],
            "layer_abs_values": layer_abs_values
        })
    
    # Clean up layer names by removing common prefix
    def clean_layer_name(name):
        # Remove common prefixes like 'base_model.model.model.'
        if name.startswith('base_model.model.model.'):
            return name[len('base_model.model.model.'):]
        return name
    
    # Sort layers by their position in the model
    sorted_layers = sorted(all_layers)
    
    # Create DataFrames for plotting
    abs_values_data = []
    pct_diff_data = []
    
    # Get reference values from first dataset
    reference_dataset = data_sets[0]
    
    for layer in sorted_layers:
        clean_name = clean_layer_name(layer)
        
        # Skip layers that don't exist in the reference dataset
        if layer not in reference_dataset["layer_abs_values"]:
            continue
        
        reference_value = reference_dataset["layer_abs_values"][layer]
        
        for dataset in data_sets:
            # Skip if this layer doesn't exist in this dataset
            if layer not in dataset["layer_abs_values"]:
                continue
                
            value = dataset["layer_abs_values"][layer]
            
            # Absolute values
            abs_values_data.append({
                "Layer": clean_name,
                "Expected Absolute Value": value,
                "Dataset": dataset["display_name"]
            })
            
            # Percentage difference
            pct_diff = ((value - reference_value) / reference_value) * 100
            pct_diff_data.append({
                "Layer": clean_name,
                "Percentage Difference (%)": pct_diff,
                "Dataset": dataset["display_name"]
            })
    
    # Create DataFrames
    abs_df = pd.DataFrame(abs_values_data)
    pct_df = pd.DataFrame(pct_diff_data)
    
    # Create color mapping for datasets
    color_map = {dataset["display_name"]: dataset["color"] for dataset in data_sets}
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Expected Absolute Values
    sns.lineplot(
        data=abs_df, 
        x="Layer", 
        y="Expected Absolute Value", 
        hue="Dataset",
        palette=color_map,
        marker='o',
        ax=ax1
    )
    ax1.set_title("Expected Absolute Values Across Layers")
    ax1.set_ylabel("Expected Absolute Value")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Percentage Differences
    sns.lineplot(
        data=pct_df, 
        x="Layer", 
        y="Percentage Difference (%)", 
        hue="Dataset",
        palette=color_map,
        marker='o',
        ax=ax2
    )
    ax2.set_title(f"Percentage Difference from {reference_dataset['display_name']}")
    ax2.set_ylabel("Percentage Difference (%)")
    ax2.set_xlabel("Layer")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line at 0% for reference
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_positive_firing_fractions(scalar_set_paths, display_names=None):
    """
    Plot the fraction of positive firing examples for each layer across multiple datasets.
    
    Args:
        scalar_set_paths: Dictionary mapping display names to paths, or list of paths
        display_names: Optional dictionary mapping file paths to display names (if scalar_set_paths is a list)
    
    Returns:
        A figure with two plots:
        1. Scatter plot showing the fraction of positive firing examples
        2. Scatter plot showing the relative fraction compared to the first dataset
    """
    # Handle different input formats
    if isinstance(scalar_set_paths, dict):
        # If dictionary provided, use it directly
        display_dict = scalar_set_paths
    else:
        # If list provided, create dictionary with optional display names
        if isinstance(scalar_set_paths, str):
            scalar_set_paths = [scalar_set_paths]
            
        if display_names is None:
            # Use filenames as display names
            display_dict = {Path(path).stem: path for path in scalar_set_paths}
        else:
            # Use provided display names
            display_dict = {display_names.get(path, Path(path).stem): path 
                           for path in scalar_set_paths}
    
    # Define colors for different files
    file_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Load data and calculate statistics
    data_sets = []
    all_layers = set()
    
    for i, (display_name, path) in enumerate(display_dict.items()):
        # Load data
        data = torch.load(path)
        
        # Create layer-wise statistics
        layer_stats = {}
        
        # Group scalars by layer
        for prompt, token_dict in data["scalar_set"].items():
            for token_idx, token_info in token_dict.items():
                for layer_name, scalar in token_info[1].items():
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = []
                    layer_stats[layer_name].append(scalar)
        
        # Calculate fraction of positive values for each layer
        layer_positive_fractions = {}
        for layer_name, scalars in layer_stats.items():
            positive_count = sum(1 for s in scalars if s > 0)
            total_count = len(scalars)
            layer_positive_fractions[layer_name] = positive_count / total_count if total_count > 0 else 0
            all_layers.add(layer_name)
        
        data_sets.append({
            "display_name": display_name,
            "color": file_colors[i % len(file_colors)],
            "layer_positive_fractions": layer_positive_fractions
        })
    
    # Clean up layer names by removing common prefix
    def clean_layer_name(name):
        # Remove common prefixes like 'base_model.model.model.'
        if name.startswith('base_model.model.model.'):
            return name[len('base_model.model.model.'):]
        return name
    
    # Sort layers by their position in the model
    sorted_layers = sorted(all_layers)
    
    # Get reference dataset (first one)
    reference_dataset = data_sets[0]
    
    # Create DataFrames for plotting
    abs_data = []
    rel_data = []
    
    for layer in sorted_layers:
        clean_name = clean_layer_name(layer)
        
        # Skip layers that don't exist in the reference dataset
        if layer not in reference_dataset["layer_positive_fractions"]:
            continue
            
        reference_fraction = reference_dataset["layer_positive_fractions"][layer]
        
        for dataset in data_sets:
            # Skip if this layer doesn't exist in this dataset
            if layer not in dataset["layer_positive_fractions"]:
                continue
                
            fraction = dataset["layer_positive_fractions"][layer]
            
            # Absolute fractions
            abs_data.append({
                "Layer": clean_name,
                "Fraction of Positive Values": fraction,
                "Dataset": dataset["display_name"]
            })
            
            # Relative fractions (compared to reference)
            if reference_fraction > 0:
                rel_fraction = fraction / reference_fraction
            else:
                rel_fraction = float('nan')  # Handle division by zero
                
            rel_data.append({
                "Layer": clean_name,
                "Relative Fraction": rel_fraction,
                "Dataset": dataset["display_name"]
            })
    
    # Create DataFrames
    abs_df = pd.DataFrame(abs_data)
    rel_df = pd.DataFrame(rel_data)
    
    # Create color mapping for datasets
    color_map = {dataset["display_name"]: dataset["color"] for dataset in data_sets}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Absolute fractions
    sns.scatterplot(
        data=abs_df, 
        x="Layer", 
        y="Fraction of Positive Values", 
        hue="Dataset",
        palette=color_map,
        s=100,  # Larger point size
        alpha=0.7,
        ax=ax1
    )
    
    # Add horizontal line at 0.5 for reference
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal positive/negative')
    
    # Customize first plot
    ax1.set_title("Fraction of Positive Firing Examples by Layer")
    ax1.set_ylabel("Fraction of Positive Values")
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Relative fractions
    sns.scatterplot(
        data=rel_df, 
        x="Layer", 
        y="Relative Fraction", 
        hue="Dataset",
        palette=color_map,
        s=100,  # Larger point size
        alpha=0.7,
        ax=ax2
    )
    
    # Add horizontal line at 1.0 for reference
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal to reference')
    
    # Customize second plot
    ax2.set_title(f"Relative Fraction of Positive Firing (Compared to {reference_dataset['display_name']})")
    ax2.set_ylabel("Relative Fraction")
    ax2.set_xlabel("Layer")
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def save_top_bottom_examples(scalar_set_paths, output_path, exclude_layers=None, n=100):
    """
    Save the top and bottom n examples across all datasets for each layer.
    
    Args:
        scalar_set_paths: Dictionary mapping display names to paths, or list of paths
        output_path: Path to save the JSON file
        n: Number of top and bottom examples to save for each layer (default: 30)
        exclude_layers: Optional list of layer name patterns to exclude
    
    Returns:
        Path to the saved JSON file
    """
    # Define token filter string to exclude system prompts and other boilerplate
    token_filter_str = "system You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>Ċ<|im_start|>userĊassistant"
    
    # Handle different input formats
    if isinstance(scalar_set_paths, dict):
        # If dictionary provided, extract paths
        paths = list(scalar_set_paths.values())
    else:
        # If list or string provided
        if isinstance(scalar_set_paths, str):
            paths = [scalar_set_paths]
        else:
            paths = scalar_set_paths
    
    # Initialize data structures
    all_layers = set()
    layer_examples = {}
    
    # Process each dataset
    for path in paths:
        # Load data
        data = torch.load(path)
        scalar_set = data["scalar_set"]
        tokenized_prompts = data["tokenized_prompts"]
        
        # Process each prompt
        for prompt_id, token_dict in scalar_set.items():
            prompt_tokens = tokenized_prompts.get(prompt_id, [])
            
            # Process each token
            for token_idx, token_info in token_dict.items():
                token_text = token_info[0]
                token_idx = int(token_idx)
                
                # Skip tokens that are part of the filter string
                if token_text in token_filter_str:
                    continue
                
                # Check if token is in a filtered context
                context_start = max(0, token_idx - 10)
                context_end = min(len(prompt_tokens), token_idx + 11)
                context_check = " ".join(prompt_tokens[context_start:context_end])
                
                # Skip if the context contains parts of the filter string
                if any(filter_part in context_check for filter_part in ["system You are", "<|im_end|>", "<|im_start|>"]):
                    continue
                
                # Get the context (sentence) around the token
                context_start = max(0, token_idx - 30)
                context_end = min(len(prompt_tokens), token_idx + 31)
                
                # Create the context string with the token highlighted
                context = []
                for i in range(context_start, context_end):
                    if i == token_idx:
                        # Highlight the token
                        context.append(f"[{prompt_tokens[i]}]")
                    else:
                        context.append(prompt_tokens[i])
                
                context_str = " ".join(context).replace(" ##", "").replace("Ġ", " ")
                
                # Process each layer's scalar
                for layer_name, scalar in token_info[1].items():
                    # Skip excluded layers
                    if exclude_layers and any(pattern in layer_name for pattern in exclude_layers):
                        continue
                    
                    # Initialize layer data if not exists
                    if layer_name not in layer_examples:
                        layer_examples[layer_name] = {
                            "top": [],
                            "bottom": []
                        }
                        all_layers.add(layer_name)
                    
                    # Add example with its scalar value
                    example = {
                        "scalar": float(scalar),
                        "context": context_str,
                        "token": token_text
                    }
                    
                    # Add to appropriate list
                    layer_examples[layer_name]["top"].append(example)
                    layer_examples[layer_name]["bottom"].append(example)
    
    # Sort and trim examples for each layer
    result = {}
    for layer_name in all_layers:
        # Sort top examples (highest scalar first)
        layer_examples[layer_name]["top"].sort(key=lambda x: x["scalar"], reverse=True)
        # Sort bottom examples (lowest scalar first)
        layer_examples[layer_name]["bottom"].sort(key=lambda x: x["scalar"])
        
        # Take only the top/bottom n
        top_examples = layer_examples[layer_name]["top"][:n]
        bottom_examples = layer_examples[layer_name]["bottom"][:n]
        
        # Clean up layer name
        clean_name = layer_name
        if layer_name.startswith('base_model.model.model.'):
            clean_name = layer_name[len('base_model.model.model.'):]
        
        # Store in result
        result[clean_name] = {
            "top": top_examples,
            "bottom": bottom_examples
        }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved top/bottom {n} examples for {len(all_layers)} layers to {output_path}")
    return output_path