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
        token_filter_str = "systemYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>Ċ<|im_start|>userĊassistant"
        df = df[~df["token"].isin([t for t in df["token"] if token_filter_str in t])]
    
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

def display_scalar_analysis_interactive(scalar_set_path):
    """
    Display an interactive visualization of the scalar set analysis in the notebook.
    
    Args:
        scalar_set_path: Path to the saved scalar set
    """
    # Load the scalar set
    data = torch.load(scalar_set_path)
    scalar_set = data["scalar_set"]
    tokenized_prompts = data["tokenized_prompts"]
    
    # Create visualization data
    visualization = create_lora_scalar_visualization(
        scalar_set_path, 
        output_path=None,  # Don't save to file
        context_window=10,
        top_k=10
    )
    
    # Create layer selector dropdown
    layers = list(visualization.keys())
    layer_dropdown = widgets.Dropdown(
        options=layers,
        description='Layer:',
        disabled=False,
    )
    
    # Create tab for top/bottom tokens
    tabs = widgets.Tab()
    tab_contents = ['Top Tokens', 'Bottom Tokens']
    
    # Function to display token contexts

    def display_token_contexts(tokens_data):
        html_output = "<div style='max-width: 800px;'>"
        for token_data in tokens_data:
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
                    # The text appears grey because opacity:0.5 is making it semi-transparent
                    # Increase opacity and ensure strong contrast with background
                    html_output += f"<span style='background-color: {color}; color: black; font-weight: bold;'>{text}</span>"
                else:
                    html_output += text
            
            html_output += "</div>"
        
        html_output += "</div>"
        return HTML(html_output)
    
    # Create distribution plot widget
    plot_output = widgets.Output()
    
    # Create output widgets for token displays
    top_tokens_output = widgets.Output()
    bottom_tokens_output = widgets.Output()
    
    # Function to update display when layer changes
    def on_layer_change(change):
        layer = change['new']
        layer_data = visualization[layer]
        
        # Update distribution plot
        with plot_output:
            plot_output.clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            
            # Create DataFrame directly instead of using get_scalar_stats
            df = pd.DataFrame([
                {
                    "prompt": prompt, 
                    "token": token_info[0], 
                    "token_idx": token_idx,
                    "layer": layer_name, 
                    "scalar": scalar
                }
                for prompt, token_dict in scalar_set.items()
                for token_idx, token_info in token_dict.items()
                for layer_name, scalar in token_info[1].items()
            ])
            
            layer_df = df[df["layer"] == layer]
            scalar_values = layer_df["scalar"].values
            
            sns.histplot(scalar_values, kde=True)
            plt.title(f"Distribution of Scalar Values for {layer}")
            plt.xlabel("Scalar Value")
            plt.ylabel("Frequency")
            
            # Add distribution stats as text
            stats = layer_data["distribution_stats"]
            plt.figtext(0.01, 0.01, 
                       f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}\nMin: {stats['min']:.3f}, Max: {stats['max']:.3f}",
                       wrap=True, fontsize=10)
            
            plt.tight_layout()
            plt.show()
        
        # Update token displays
        with top_tokens_output:
            top_tokens_output.clear_output(wait=True)
            display(display_token_contexts(layer_data["top_tokens"]))
            
        with bottom_tokens_output:
            bottom_tokens_output.clear_output(wait=True)
            display(display_token_contexts(layer_data["bottom_tokens"]))
    
    # Set up tabs
    tabs.children = [top_tokens_output, bottom_tokens_output]
    for i, title in enumerate(tab_contents):
        tabs.set_title(i, title)
    
    # Set up the layout
    layer_dropdown.observe(on_layer_change, names='value')
    
    # Initialize with first layer
    if layers:
        on_layer_change({'new': layers[0]})
    
    # Display widgets
    display(widgets.VBox([
        widgets.HTML("<h2>LoRA Scalar Analysis</h2>"),
        layer_dropdown,
        plot_output,
        widgets.HTML("<h3>Token Analysis</h3>"),
        tabs
    ]))