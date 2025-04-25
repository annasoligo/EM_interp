"""
Utility functions for generating interactive dashboards for adaptor analysis.
This module provides tools to visualize and explore adaptor values across tokens.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union, Optional


def preprocess_adaptor_data(data: Dict) -> Tuple[Dict, List, List, Dict, Dict]:
    """
    Process the raw adaptor data into a format suitable for visualization.
    
    Args:
        data: The adaptor data dictionary with prompt, tokens, and adaptor values
    
    Returns:
        Tuple containing:
        - processed_data: Clean data structure
        - tokens: List of token strings
        - adaptors: List of adaptor names (simplified)
        - token_ranges: Min/max values for each token
        - adaptor_ranges: Min/max values for each adaptor
    """
    # Extract first prompt (assuming only one prompt in the data)
    prompt = list(data.keys())[0]
    processed_data = data[prompt]
    
    # Extract tokens and create clean token list (remove special chars if needed)
    tokens = [processed_data[idx][0] for idx in sorted([int(i) for i in processed_data.keys()])]
    clean_tokens = [t.replace('Ġ', ' ').strip() if t.startswith('Ġ') else t for t in tokens]
    
    # Extract adaptor names and create simplified versions
    adaptors = list(processed_data[0][1].keys())
    simplified_adaptors = []
    for adaptor in adaptors:
        match = adaptor.split('layers.')[1].split('.mlp')[0]
        simplified_adaptors.append(f"down_proj {match}")
    
    # Calculate ranges for normalization
    token_ranges = {}
    for idx, (token, values) in processed_data.items():
        values_list = list(values.values())
        token_ranges[token] = {
            'min': min(values_list),
            'max': max(values_list)
        }
    
    adaptor_ranges = {}
    for adaptor in adaptors:
        values = [processed_data[idx][1][adaptor] for idx in processed_data]
        adaptor_ranges[adaptor] = {
            'min': min(values),
            'max': max(values)
        }
    
    # Convert processed_data for easier access
    clean_data = {}
    for idx, (token, values) in processed_data.items():
        clean_data[token] = values
    
    return clean_data, tokens, simplified_adaptors, token_ranges, adaptor_ranges


def create_value_matrix(data: Dict, tokens: List, adaptors: List) -> np.ndarray:
    """
    Create a matrix of values for heatmap visualization.
    
    Args:
        data: Processed data dictionary
        tokens: List of token strings
        adaptors: List of adaptor full names (not simplified)
    
    Returns:
        matrix: numpy array with shape (len(tokens), len(adaptors))
    """
    matrix = np.zeros((len(tokens), len(adaptors)))
    
    for i, token in enumerate(tokens):
        token_idx = next(idx for idx, (tok, _) in data.items() if tok == token)
        for j, adaptor in enumerate(adaptors):
            matrix[i, j] = data[token_idx][1][adaptor]
    
    return matrix


def create_adaptor_dashboard(data: Dict):
    """
    Create an interactive dashboard for exploring adaptor values across tokens.
    
    Args:
        data: The adaptor data dictionary
    """
    # Process the data
    prompt = list(data.keys())[0]
    raw_data = data[prompt]
    
    # Convert all keys to strings for consistency
    raw_data_normalized = {}
    for key, value in raw_data.items():
        raw_data_normalized[str(key)] = value
    
    # Get the total number of tokens
    num_tokens = len(raw_data)
    
    # Extract tokens and adaptors - handle both string and integer keys
    tokens = []
    for idx in range(num_tokens):
        # Try both string and integer keys
        if str(idx) in raw_data:
            tokens.append(raw_data[str(idx)][0])
        elif idx in raw_data:
            tokens.append(raw_data[idx][0])
        else:
            raise KeyError(f"Cannot find token at index {idx} in data")
    
    clean_tokens = [t.replace('Ġ', ' ').strip() if t.startswith('Ġ') else t for t in tokens]
    
    # Get the first token's adaptor keys - try both string and integer keys
    first_token_data = None
    if '0' in raw_data:
        first_token_data = raw_data['0']
    elif 0 in raw_data:
        first_token_data = raw_data[0]
    else:
        raise KeyError("Cannot find first token (index 0) in data")
        
    adaptors = list(first_token_data[1].keys())
    
    # Create simplified adaptor names for display
    simplified_adaptors = []
    for adaptor in adaptors:
        layer_num = adaptor.split('layers.')[1].split('.mlp')[0]
        simplified_adaptors.append(f"down_proj {layer_num}")
    
    # Create a matrix of values
    value_matrix = np.zeros((len(tokens), len(adaptors)))
    for i in range(len(tokens)):
        # Try both string and integer keys
        if str(i) in raw_data:
            token_data = raw_data[str(i)]
        elif i in raw_data:
            token_data = raw_data[i]
        else:
            raise KeyError(f"Cannot find token at index {i} in data")
            
        for j, adaptor in enumerate(adaptors):
            value_matrix[i, j] = token_data[1][adaptor]
    
    # Calculate global min/max for normalization
    global_min = np.min(value_matrix)
    global_max = np.max(value_matrix)
    
    # Store token-specific min/max
    token_ranges = {}
    for i, token in enumerate(tokens):
        token_ranges[token] = {
            'min': np.min(value_matrix[i, :]),
            'max': np.max(value_matrix[i, :])
        }
    
    # Store adaptor-specific min/max
    adaptor_ranges = {}
    for j, adaptor in enumerate(adaptors):
        adaptor_ranges[adaptor] = {
            'min': np.min(value_matrix[:, j]),
            'max': np.max(value_matrix[:, j])
        }
    
    # Create visualization widgets
    token_dropdown = widgets.Dropdown(
        options=clean_tokens,
        description='Token:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    adaptor_dropdown = widgets.Dropdown(
        options=simplified_adaptors,
        description='Adaptor:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    normalization_radio = widgets.RadioButtons(
        options=['Global', 'Per Token', 'Per Adaptor'],
        description='Normalize:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )
    
    # Add view selection for cleaner display
    view_selection = widgets.RadioButtons(
        options=['Heatmap + Token + Adaptor', 'Heatmap + Token', 'Heatmap + Adaptor', 'Heatmap Only'],
        description='View:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='250px')
    )
    
    # Add token pagination controls for long sequences
    token_group_size = min(20, len(clean_tokens))  # Default to show 20 tokens at a time, or all if less than 20
    
    # Calculate number of token pages
    token_pages = max(1, (len(clean_tokens) + token_group_size - 1) // token_group_size)
    token_page_options = [f"Page {i+1} (tokens {i*token_group_size+1}-{min((i+1)*token_group_size, len(clean_tokens))})" 
                          for i in range(token_pages)]
    
    token_page_dropdown = widgets.Dropdown(
        options=token_page_options,
        value=token_page_options[0] if token_page_options else None,
        description='Token Page:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    # Create a slider for token group size
    token_group_size_slider = widgets.IntSlider(
        value=token_group_size,
        min=5,
        max=min(50, len(clean_tokens)),
        step=5,
        description='Tokens per page:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )
    
    # Add figure size controls for better visualization
    figure_height_slider = widgets.IntSlider(
        value=9,
        min=5,
        max=20,
        step=1,
        description='Figure height:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px')
    )
    
    # Checkbox to enable scrollable output
    scrollable_output = widgets.Checkbox(
        value=True,
        description='Scrollable output',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )
    
    # Output widget for displaying plots
    output = widgets.Output()
    
    # Define plot functions
    def plot_token_adaptors(token, normalization):
        """Plot adaptor values for a specific token"""
        token_idx = clean_tokens.index(token)
        original_token = tokens[token_idx]
        
        values = value_matrix[token_idx, :]
        
        if normalization == 'Global':
            vmin, vmax = global_min, global_max
        elif normalization == 'Per Token':
            vmin, vmax = token_ranges[original_token]['min'], token_ranges[original_token]['max']
        else:  # Per Adaptor
            vmin, vmax = None, None
            # We'll normalize each bar individually
        
        plt.figure(figsize=(14, 7))
        
        # Normalize per adaptor if needed
        if normalization == 'Per Adaptor':
            normalized_values = []
            for j, adaptor in enumerate(adaptors):
                adaptor_min = adaptor_ranges[adaptor]['min']
                adaptor_max = adaptor_ranges[adaptor]['max']
                if adaptor_max != adaptor_min:
                    norm_val = (values[j] - adaptor_min) / (adaptor_max - adaptor_min)
                else:
                    norm_val = 0.5
                normalized_values.append(norm_val)
            
            # Create a heatmap-style horizontal bar chart
            plt.barh(simplified_adaptors, normalized_values, color=plt.cm.RdBu_r(normalized_values))
            plt.title(f'Normalized LoRA Scalars (αA^Tx_h) for Token "{token}" (Per Adaptor)', fontsize=18)
            plt.xlim(0, 1)
            plt.xlabel('Normalized Value (0-1)', fontsize=16)
        else:
            # Use the actual values with a colormap
            colors = plt.cm.RdBu_r((values - vmin) / (vmax - vmin))
            plt.barh(simplified_adaptors, values, color=colors)
            plt.title(f'LoRA Scalars (αA^Tx_h) for Token "{token}"', fontsize=18)
            plt.xlabel('Value', fontsize=16)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.ylabel('Adaptor', fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.tight_layout()
    
    def plot_adaptor_tokens(adaptor, normalization, token_page=0, tokens_per_page=20, fig_height=7):
        """Plot token values for a specific adaptor"""
        adaptor_idx = simplified_adaptors.index(adaptor)
        original_adaptor = adaptors[adaptor_idx]
        
        # Get all values for this adaptor
        values = value_matrix[:, adaptor_idx]
        
        # Calculate token slice for pagination
        start_idx = token_page * tokens_per_page
        end_idx = min(start_idx + tokens_per_page, len(tokens))
        
        # Slice the data for the current token page
        values_slice = values[start_idx:end_idx]
        tokens_slice = clean_tokens[start_idx:end_idx]
        
        if normalization == 'Global':
            vmin, vmax = global_min, global_max
        elif normalization == 'Per Adaptor':
            vmin, vmax = adaptor_ranges[original_adaptor]['min'], adaptor_ranges[original_adaptor]['max']
        else:  # Per Token
            vmin, vmax = None, None
            # We'll normalize each bar individually 
        
        # Adjust figure size based on number of tokens shown
        tokens_width_factor = min(0.7 * tokens_per_page, 14)  # Limit max width
        plt.figure(figsize=(tokens_width_factor, fig_height))
        
        # Normalize per token if needed
        if normalization == 'Per Token':
            normalized_values = []
            for i in range(start_idx, end_idx):
                token = tokens[i]
                token_min = token_ranges[token]['min']
                token_max = token_ranges[token]['max']
                if token_max != token_min:
                    norm_val = (values[i] - token_min) / (token_max - token_min)
                else:
                    norm_val = 0.5
                normalized_values.append(norm_val)
            
            # Create vertical bar chart with tokens on x-axis
            plt.bar(tokens_slice, normalized_values, color=plt.cm.RdBu_r(normalized_values))
            
            # Calculate page information for title
            page_info = f" (Page {token_page+1}/{(len(tokens) + tokens_per_page - 1) // tokens_per_page}, Tokens {start_idx+1}-{end_idx})"
            plt.title(f'Normalized LoRA Scalars (αA^Tx_h) for Adaptor "{adaptor}" (Per Token){page_info}', fontsize=18)
            plt.ylim(0, 1)
            plt.ylabel('Normalized Value (0-1)', fontsize=16)
        else:
            # Use the actual values with a colormap
            norm_values = (values_slice - vmin) / (vmax - vmin) if vmax != vmin else np.ones_like(values_slice) * 0.5
            colors = plt.cm.RdBu_r(norm_values)
            plt.bar(tokens_slice, values_slice, color=colors)
            
            # Calculate page information for title
            page_info = f" (Page {token_page+1}/{(len(tokens) + tokens_per_page - 1) // tokens_per_page}, Tokens {start_idx+1}-{end_idx})"
            plt.title(f'LoRA Scalars (αA^Tx_h) for Adaptor "{adaptor}"{page_info}', fontsize=18)
            plt.ylabel('Value', fontsize=16)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('Token', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
    
    def create_heatmap(normalization, token_page=0, tokens_per_page=20, fig_height=9):
        """Create a heatmap of all token-adaptor values"""
        if normalization == 'Global':
            vmin, vmax = global_min, global_max
            norm = None
        elif normalization == 'Per Token':
            # Normalize each row (token)
            norm_data = np.zeros_like(value_matrix)
            for i, token in enumerate(tokens):
                token_min = token_ranges[token]['min']
                token_max = token_ranges[token]['max']
                if token_max != token_min:
                    norm_data[i, :] = (value_matrix[i, :] - token_min) / (token_max - token_min)
                else:
                    norm_data[i, :] = 0.5
            value_matrix_norm = norm_data
            vmin, vmax = 0, 1
            norm = None
        elif normalization == 'Per Adaptor':
            # Normalize each column (adaptor)
            norm_data = np.zeros_like(value_matrix)
            for j, adaptor in enumerate(adaptors):
                adaptor_min = adaptor_ranges[adaptor]['min']
                adaptor_max = adaptor_ranges[adaptor]['max']
                if adaptor_max != adaptor_min:
                    norm_data[:, j] = (value_matrix[:, j] - adaptor_min) / (adaptor_max - adaptor_min)
                else:
                    norm_data[:, j] = 0.5
            value_matrix_norm = norm_data
            vmin, vmax = 0, 1
            norm = None
        
        # Use the normalized data if needed
        data_to_plot = value_matrix_norm if normalization != 'Global' else value_matrix
        
        # Calculate token slice for pagination
        start_idx = token_page * tokens_per_page
        end_idx = min(start_idx + tokens_per_page, len(tokens))
        
        # Slice the data for the current token page
        data_slice = data_to_plot[start_idx:end_idx, :]
        tokens_slice = clean_tokens[start_idx:end_idx]
        
        # Adjust figure size based on number of tokens shown and user preference
        tokens_width_factor = min(0.7 * tokens_per_page, 14)  # Limit max width
        plt.figure(figsize=(tokens_width_factor, fig_height))
        
        # Transpose the data for swapped axes (tokens on x-axis, adaptors on y-axis)
        data_to_plot_swapped = data_slice.T
        
        # Create the heatmap with swapped axes
        sns.heatmap(data_to_plot_swapped, 
                   yticklabels=simplified_adaptors,  # Now y-axis has adaptors
                   xticklabels=tokens_slice,        # Now x-axis has tokens (paginated)
                   cmap='RdBu_r',
                   center=0 if normalization == 'Global' else 0.5,
                   vmin=vmin,
                   vmax=vmax,
                   annot=False,  # Set to True if you want to see values inside cells
                   fmt='.2f')    # Format for annotations if enabled
        
        # Calculate page information for title
        page_info = f" (Page {token_page+1}/{(len(tokens) + tokens_per_page - 1) // tokens_per_page}, Tokens {start_idx+1}-{end_idx})"
        
        plt.title(f'Per Token LoRA Scalars (αA^Tx_h) ({normalization} Normalization){page_info}', fontsize=18)
        plt.xlabel('Tokens', fontsize=16)
        plt.ylabel('Adaptors', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
    
    # Function to update token pagination controls
    def update_token_pagination(*args):
        # Get the new token group size
        new_group_size = token_group_size_slider.value
        
        # Calculate new number of pages
        new_pages = max(1, (len(clean_tokens) + new_group_size - 1) // new_group_size)
        
        # Create new page options
        new_page_options = [f"Page {i+1} (tokens {i*new_group_size+1}-{min((i+1)*new_group_size, len(clean_tokens))})" 
                           for i in range(new_pages)]
        
        # Update the dropdown options
        token_page_dropdown.options = new_page_options
        
        # Set to first page if current value doesn't exist in new options
        if token_page_dropdown.value not in new_page_options and new_page_options:
            token_page_dropdown.value = new_page_options[0]
        
        # Trigger plot update
        update_plot()
    
    # Connect token group size slider to pagination update function
    token_group_size_slider.observe(update_token_pagination, names='value')
    
    # Define update function for interactive widgets
    def update_plot(*args):
        with output:
            clear_output(wait=True)
            
            # Get the selected view and pagination settings
            selected_view = view_selection.value
            
            # Parse the current token page from dropdown
            if token_page_dropdown.value:
                current_page = int(token_page_dropdown.value.split(' ')[1].split(' ')[0]) - 1
            else:
                current_page = 0
                
            # Get tokens per page and figure height
            tokens_per_page = token_group_size_slider.value
            fig_height = figure_height_slider.value
            
            # Update output container scrollability
            output_container.layout.overflow = 'auto' if scrollable_output.value else 'visible'
            
            # Always create the heatmap first (now always at the top)
            create_heatmap(normalization_radio.value, current_page, tokens_per_page, fig_height)
            plt.show()
            
            # Create the other plots based on view selection
            if "Token" in selected_view and "Adaptor" in selected_view:
                # Show both token and adaptor plots side by side
                fig = plt.figure(figsize=(20, fig_height))
                
                # Token plot - this doesn't need pagination since it shows one token's values across adaptors
                plt.subplot(1, 2, 1)
                token_idx = clean_tokens.index(token_dropdown.value)
                original_token = tokens[token_idx]
                values = value_matrix[token_idx, :]
                
                if normalization_radio.value == 'Global':
                    vmin, vmax = global_min, global_max
                    colors = plt.cm.RdBu_r((values - vmin) / (vmax - vmin))
                    plt.bar(simplified_adaptors, values, color=colors)
                    plt.title(f'LoRA Scalars for Token "{token_dropdown.value}"', fontsize=18)
                    plt.ylabel('Value', fontsize=16)
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                else:
                    # Use normalized values
                    norm_values = []
                    for j, adaptor in enumerate(adaptors):
                        if normalization_radio.value == 'Per Token':
                            token_min = token_ranges[original_token]['min']
                            token_max = token_ranges[original_token]['max']
                            norm_val = (values[j] - token_min) / (token_max - token_min) if token_max != token_min else 0.5
                        else:  # Per Adaptor
                            adaptor_min = adaptor_ranges[adaptor]['min']
                            adaptor_max = adaptor_ranges[adaptor]['max']
                            norm_val = (values[j] - adaptor_min) / (adaptor_max - adaptor_min) if adaptor_max != adaptor_min else 0.5
                        norm_values.append(norm_val)
                    
                    plt.bar(simplified_adaptors, norm_values, color=plt.cm.RdBu_r(norm_values))
                    plt.title(f'Normalized LoRA Scalars for Token "{token_dropdown.value}"', fontsize=18)
                    plt.ylabel('Normalized Value', fontsize=16)
                
                plt.xlabel('Adaptor', fontsize=16)
                plt.xticks(rotation=45, ha='right', fontsize=14)
                plt.yticks(fontsize=14)
                
                # Adaptor plot with pagination
                plt.subplot(1, 2, 2)
                # Use paginated adaptor plot function
                adaptor_idx = simplified_adaptors.index(adaptor_dropdown.value)
                original_adaptor = adaptors[adaptor_idx]
                values = value_matrix[:, adaptor_idx]
                
                # Calculate token slice for pagination
                start_idx = current_page * tokens_per_page
                end_idx = min(start_idx + tokens_per_page, len(tokens))
                values_slice = values[start_idx:end_idx]
                tokens_slice = clean_tokens[start_idx:end_idx]
                
                if normalization_radio.value == 'Global':
                    vmin, vmax = global_min, global_max
                    colors = plt.cm.RdBu_r((values_slice - vmin) / (vmax - vmin))
                    plt.bar(tokens_slice, values_slice, color=colors)
                    page_info = f" (Page {current_page+1}/{(len(tokens) + tokens_per_page - 1) // tokens_per_page})"
                    plt.title(f'LoRA Scalars for Adaptor "{adaptor_dropdown.value}"{page_info}', fontsize=18)
                    plt.ylabel('Value', fontsize=16)
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                else:
                    # Use normalized values
                    norm_values = []
                    for i in range(start_idx, end_idx):
                        token = tokens[i]
                        if normalization_radio.value == 'Per Adaptor':
                            adaptor_min = adaptor_ranges[original_adaptor]['min']
                            adaptor_max = adaptor_ranges[original_adaptor]['max']
                            norm_val = (values[i] - adaptor_min) / (adaptor_max - adaptor_min) if adaptor_max != adaptor_min else 0.5
                        else:  # Per Token
                            token_min = token_ranges[token]['min']
                            token_max = token_ranges[token]['max']
                            norm_val = (values[i] - token_min) / (token_max - token_min) if token_max != token_min else 0.5
                        norm_values.append(norm_val)
                    
                    plt.bar(tokens_slice, norm_values, color=plt.cm.RdBu_r(norm_values))
                    page_info = f" (Page {current_page+1}/{(len(tokens) + tokens_per_page - 1) // tokens_per_page})"
                    plt.title(f'Normalized LoRA Scalars for Adaptor "{adaptor_dropdown.value}"{page_info}', fontsize=18)
                    plt.ylabel('Normalized Value', fontsize=16)
                
                plt.xlabel('Token', fontsize=16)
                plt.xticks(rotation=45, ha='right', fontsize=14)
                plt.yticks(fontsize=14)
                
                plt.tight_layout()
                plt.show()
                
            elif "Token" in selected_view:
                # Single token plot - shows one token's values across adaptors (no pagination needed)
                token_idx = clean_tokens.index(token_dropdown.value)
                original_token = tokens[token_idx]
                values = value_matrix[token_idx, :]
                
                plt.figure(figsize=(14, fig_height))
                
                if normalization_radio.value == 'Per Adaptor':
                    norm_values = []
                    for j, adaptor in enumerate(adaptors):
                        adaptor_min = adaptor_ranges[adaptor]['min']
                        adaptor_max = adaptor_ranges[adaptor]['max']
                        if adaptor_max != adaptor_min:
                            norm_val = (values[j] - adaptor_min) / (adaptor_max - adaptor_min)
                        else:
                            norm_val = 0.5
                        norm_values.append(norm_val)
                    
                    plt.bar(simplified_adaptors, norm_values, color=plt.cm.RdBu_r(norm_values))
                    plt.title(f'Normalized LoRA Scalars for Token "{token_dropdown.value}" (Per Adaptor)', fontsize=18)
                    plt.ylim(0, 1)
                    plt.ylabel('Normalized Value (0-1)', fontsize=16)
                else:
                    if normalization_radio.value == 'Global':
                        vmin, vmax = global_min, global_max
                    else:  # Per Token
                        vmin, vmax = token_ranges[original_token]['min'], token_ranges[original_token]['max']
                    
                    colors = plt.cm.RdBu_r((values - vmin) / (vmax - vmin))
                    plt.bar(simplified_adaptors, values, color=colors)
                    plt.title(f'LoRA Scalars for Token "{token_dropdown.value}"', fontsize=18)
                    plt.ylabel('Value', fontsize=16)
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                plt.xlabel('Adaptor', fontsize=16)
                plt.xticks(rotation=45, ha='right', fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                plt.show()
                
            elif "Adaptor" in selected_view:
                # Show adaptor-specific plot with tokens on x-axis (with pagination)
                plot_adaptor_tokens(adaptor_dropdown.value, normalization_radio.value, current_page, tokens_per_page, fig_height)
                plt.show()
    
    # Connect widgets to the update function
    token_dropdown.observe(update_plot, names='value')
    adaptor_dropdown.observe(update_plot, names='value')
    normalization_radio.observe(update_plot, names='value')
    view_selection.observe(update_plot, names='value')
    token_page_dropdown.observe(update_plot, names='value')
    figure_height_slider.observe(update_plot, names='value')
    scrollable_output.observe(update_plot, names='value')
    
    # Connect widgets to the update function
    token_dropdown.observe(update_plot, names='value')
    adaptor_dropdown.observe(update_plot, names='value')
    normalization_radio.observe(update_plot, names='value')
    view_selection.observe(update_plot, names='value')
    
    # Create the dashboard layout with controls organized in rows
    # First row: Main controls
    main_controls = widgets.HBox([
        token_dropdown,
        adaptor_dropdown,
        normalization_radio,
        view_selection
    ])
    
    # Second row: Pagination and visualization controls
    pagination_controls = widgets.HBox([
        token_page_dropdown,
        token_group_size_slider,
        figure_height_slider,
        scrollable_output
    ])
    
    # Make the output area scrollable if requested
    output_container = widgets.Box(
        children=[output],
        layout=widgets.Layout(
            overflow='auto',
            max_height='800px',
            border='1px solid #ddd',
            margin='5px 0px'
        )
    )
    
    # Combine all controls and output
    dashboard = widgets.VBox([
        main_controls,
        pagination_controls,
        output_container
    ])
    
    # Initial plot
    update_plot()
    
    return dashboard


def parse_adaptor_data(data_dict: Dict) -> Dict:
    """
    Parse the raw adaptor data into a clean format.
    
    Args:
        data_dict: Raw data dictionary from the model
    
    Returns:
        Clean data structure for visualization
    """
    # Extract the prompt
    prompt = list(data_dict.keys())[0]
    tokens_data = data_dict[prompt]
    
    # Convert to the expected format if needed
    result = {prompt: {}}
    
    for token_idx, token_data in tokens_data.items():
        # Convert to string key for consistency
        str_idx = str(token_idx)
        
        # If token_data is already a tuple/list with [token_text, values_dict]
        if isinstance(token_data, (list, tuple)) and len(token_data) == 2:
            result[prompt][str_idx] = token_data
        # Handle tuple with extra parentheses - common in Python->JSON conversion
        elif isinstance(token_data, tuple) and len(token_data) == 2:
            result[prompt][str_idx] = list(token_data)
        # If token_data is in another format, try to convert it
        else:
            # We'd need to customize this based on input format
            raise ValueError(f"Unexpected token data format: {token_data}")
    
    return result


def create_interactive_dashboard(data: Dict) -> widgets.VBox:
    """
    Creates and returns an interactive dashboard for visualizing adaptor values.
    
    Args:
        data: The data dictionary containing token and adaptor values
    
    Returns:
        A Jupyter widget containing the interactive dashboard
    """
    # Parse the data if needed
    try:
        parsed_data = parse_adaptor_data(data)
    except ValueError:
        # If parsing fails, assume data is already in correct format
        parsed_data = data
    
    # Create the dashboard
    return create_adaptor_dashboard(parsed_data)