import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from em_interp.vis.quadrant_plots import analyze_quadrant_percentages

def extract_layers_from_filename(filename):
    """Extract ablated layer numbers from filename."""
    try:
        # Check for the standard pattern with ablated_loras_ prefix
        if 'ablated_loras_' in filename:
            # Extract the layer range from the filename
            layer_part = filename.split('ablated_loras_')[1].split('.csv')[0]
            
            # Check for hyphen pattern (e.g., "36-47")
            if '-' in layer_part:
                try:
                    start, end = map(int, layer_part.split('-'))
                    return list(range(start, end + 1))
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse hyphenated layer range from '{layer_part}' in file '{filename}'")
            
            # Check for underscore pattern (e.g., "36_47")
            elif '_' in layer_part:
                try:
                    ranges = layer_part.split('_')
                    start = int(ranges[0])
                    end = int(ranges[1])
                    return list(range(start, end + 1))
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse underscored layer range from '{layer_part}' in file '{filename}'")
            
            # Handle single layer case
            else:
                try:
                    return [int(layer_part)]
                except ValueError:
                    print(f"Warning: Could not parse layer number from '{layer_part}' in file '{filename}'")
        
        # Try alternative patterns for layer extraction
        import re
        
        # First, look for patterns like "layer6-12", "layers6-12", "l6-12", "layer6_12", etc.
        for pattern in ['layer', 'layers', 'l']:
            if pattern in filename.lower():
                try:
                    # Extract digits after the pattern
                    parts = filename.lower().split(pattern)[1].split('.')[0]
                    
                    # Check for hyphen format
                    if '-' in parts:
                        start, end = map(int, parts.split('-'))
                        return list(range(start, end + 1))
                    
                    # Check for underscore format
                    elif '_' in parts:
                        start, end = map(int, parts.split('_'))
                        return list(range(start, end + 1))
                    
                    # Single number
                    else:
                        # Extract all digits from the part
                        digits = ''.join([c for c in parts if c.isdigit()])
                        if digits:
                            return [int(digits)]
                except (ValueError, IndexError):
                    continue  # Try next pattern
        
        # Look for any range-like patterns in the filename: both hyphen and underscore
        # This is a more general approach as a fallback
        patterns = [
            r'(\d+)-(\d+)',  # Match patterns like "36-47"
            r'(\d+)_(\d+)',  # Match patterns like "36_47"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            if matches:
                for start_str, end_str in matches:
                    try:
                        start, end = int(start_str), int(end_str)
                        # Reasonable range check (assuming we don't have more than 100 layers)
                        if 0 <= start <= end <= 100:
                            return list(range(start, end + 1))
                    except ValueError:
                        continue
            
        print(f"Could not extract layer information from filename: {filename}")
        return None
    except Exception as e:
        print(f"Error extracting layers from '{filename}': {e}")
        return None

def compute_total_misaligned(row):
    """Compute Total Misaligned percentage from a DataFrame row."""
    try:
        # Check if 'Total Misaligned' already exists
        if 'Total Misaligned' in row.index:
            return float(row['Total Misaligned'].strip('%'))
        
        # Calculate from Misaligned & Coherent and Misaligned & Incoherent
        misaligned_coherent = float(row['Misaligned & Coherent'].strip('%'))
        misaligned_incoherent = float(row['Misaligned & Incoherent'].strip('%'))
        return misaligned_coherent + misaligned_incoherent
    except (ValueError, KeyError) as e:
        print(f"Error computing Total Misaligned: {e}")
        return 0.0

def find_group_directory(base_dir, group_size):
    """
    Find the appropriate directory for a specific group size.
    Checks for directories containing the group size in their name.
    
    Args:
        base_dir: Parent directory to search in
        group_size: Size of layer groups to look for
    
    Returns:
        Path to the group-specific directory, or base_dir if none found
    """
    # First, try direct match with group size
    group_dir_candidates = []
    
    # Look for directories containing the group size in their name
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and f"group{group_size}" in item.lower():
            return item_path
        elif os.path.isdir(item_path) and f"group_{group_size}" in item.lower():
            return item_path
        elif os.path.isdir(item_path) and f"group-{group_size}" in item.lower():
            return item_path
        # Add more flexible matching for group directory names
        elif os.path.isdir(item_path) and str(group_size) in item:
            group_dir_candidates.append(item_path)
    
    # If we found any candidates, return the first one
    if group_dir_candidates:
        return group_dir_candidates[0]
    
    # Check if the base_dir itself contains CSV files with the right group size
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    has_matching_files = False
    
    for file in csv_files:
        layers = extract_layers_from_filename(file)
        if layers is not None and len(layers) == group_size:
            has_matching_files = True
            break
    
    if has_matching_files:
        return base_dir
    
    # As a fallback, return the base directory with a warning
    print(f"Warning: Could not find a specific directory for group size {group_size}. Using {base_dir}.")
    return base_dir

def plot_ablation_results(base_dir, group_size=6):
    """
    Plot ablation results showing misaligned and coherent percentages.
    
    Args:
        base_dir: Directory containing ablation results
        group_size: Size of layer groups (6 or 12)
    """
    # Find the appropriate directory for this group size
    group_dir = find_group_directory(base_dir, group_size)
    print(f"\nUsing directory for group size {group_size}: {group_dir}")
    
    # Get all CSV files
    files = [f for f in os.listdir(group_dir) if f.endswith('.csv')]
    print(f"\nFound CSV files: {files}")
    
    if not files:
        print(f"No CSV files found in {group_dir}")
        return
    
    # First, let's analyze all files at once to get the data
    all_results_df = analyze_quadrant_percentages(
        path=group_dir,
        medical_column=True,
        ignore_json=True
    )
    
    if all_results_df is None or all_results_df.empty:
        print("No results data available from analyze_quadrant_percentages")
        return
        
    print(f"\nAll results DataFrame columns: {all_results_df.columns}")
    print(f"\nAll results DataFrame:\n{all_results_df}")
    
    # Look for baseline file in both parent directory and current directory
    # First try to find baseline in current directory
    baseline_files = [f for f in files if 'baseline' in f.lower()]
    
    # If no baseline in current directory, check parent directory
    if not baseline_files:
        try:
            parent_dir = os.path.dirname(group_dir)
            parent_files = [f for f in os.listdir(parent_dir) if f.endswith('.csv') and 'baseline' in f.lower()]
            if parent_files:
                # Analyze the parent directory data to get baseline values
                parent_results_df = analyze_quadrant_percentages(
                    path=parent_dir,
                    medical_column=True,
                    ignore_json=True
                )
                if parent_results_df is not None and not parent_results_df.empty:
                    baseline_file = parent_files[0]
                    print(f"\nFound baseline file in parent directory: {baseline_file}")
                    
                    # If model_name is the index, reset it
                    if parent_results_df.index.name == 'model_name':
                        parent_results_df = parent_results_df.reset_index()
                    
                    # Add file column if it doesn't exist
                    if 'file' not in parent_results_df.columns:
                        parent_results_df['file'] = parent_results_df['model_name'].apply(
                            lambda x: next((f for f in parent_files if x in f or f.replace('.csv', '') == x), None)
                        )
                    
                    # Find baseline row
                    baseline_row = parent_results_df[parent_results_df['file'] == baseline_file]
                    if baseline_row.empty:
                        baseline_model = baseline_file.replace('.csv', '')
                        baseline_row = parent_results_df[parent_results_df['model_name'] == baseline_model]
                    
                    if not baseline_row.empty:
                        baseline_coherent = float(baseline_row['Misaligned & Coherent'].iloc[0].strip('%'))
                        print(f"Found baseline in parent directory. Coherent={baseline_coherent}")
                    else:
                        print("Baseline file found in parent directory but couldn't match with results.")
                        baseline_files = []
                else:
                    print("Failed to analyze parent directory for baseline.")
                    baseline_files = []
            else:
                baseline_files = []
        except Exception as e:
            print(f"Error checking parent directory for baseline: {e}")
            baseline_files = []
    
    # If still no baseline, use the first file
    if not baseline_files:
        print("No baseline file found. Using the first file as baseline.")
        baseline_file = files[0]
    else:
        baseline_file = baseline_files[0]
        
    print(f"\nBaseline file: {baseline_file}")
    
    # Check if model_name is the index
    if all_results_df.index.name == 'model_name':
        # If model_name is the index, we need to reset it to be able to filter by filename
        all_results_df = all_results_df.reset_index()
        
    # Add file column if it doesn't exist
    if 'file' not in all_results_df.columns:
        # Try to extract model_name from the file names
        all_results_df['file'] = all_results_df['model_name'].apply(
            lambda x: next((f for f in files if x in f or f.replace('.csv', '') == x), None)
        )
    
    # If baseline is from current directory, get values
    if not 'baseline_coherent' in locals():
        baseline_row = all_results_df[all_results_df['file'] == baseline_file]
        if baseline_row.empty:
            print(f"Warning: Baseline file {baseline_file} not found in results")
            # Try to match by model_name if file matching failed
            baseline_model = baseline_file.replace('.csv', '')
            baseline_row = all_results_df[all_results_df['model_name'] == baseline_model]
            if baseline_row.empty:
                print(f"Warning: Could not find baseline in results at all. Trying first row as baseline.")
                if len(all_results_df) > 0:
                    baseline_row = all_results_df.iloc[[0]]
                else:
                    print("No results data available.")
                    return
        
        # Extract values from the baseline
        baseline_coherent = float(baseline_row['Misaligned & Coherent'].iloc[0].strip('%'))
    
    print(f"Baseline Misaligned & Coherent: {baseline_coherent}%")
    
    # Process ablation files
    results = []
    for file in files:
        if 'baseline' in file.lower():
            continue
            
        # Try to extract layer information
        layers = extract_layers_from_filename(file)
        print(f"\nProcessing file: {file}")
        print(f"Extracted layers: {layers}")
        
        if layers is None:
            print(f"Skipping - could not extract layer information")
            continue
        
        # Check if the number of layers matches the expected group size
        # But be more flexible - allow files if we can extract layer info
        if len(layers) != group_size:
            print(f"Note: File {file} has {len(layers)} layers, which differs from expected group size {group_size}")
            # Continue processing if we're within a reasonable range
            if len(layers) not in [6, 12]:
                print(f"Skipping - layer count {len(layers)} is not a standard group size")
                continue
        
        # Store both start and end layers for plotting
        start_layer = min(layers)
        end_layer = max(layers)
        
        # Use the layer range as the x-axis label
        layer_range = f"{start_layer}-{end_layer}"
            
        # Find this file in the all_results_df
        file_row = all_results_df[all_results_df['file'] == file]
        if file_row.empty:
            # Try with model_name
            model_name = file.replace('.csv', '')
            file_row = all_results_df[all_results_df['model_name'] == model_name]
            if file_row.empty:
                print(f"Warning: File {file} not found in results")
                continue
        
        # Extract only the Misaligned & Coherent value
        coherent_value = float(file_row['Misaligned & Coherent'].iloc[0].strip('%'))
            
        results.append({
            'Start Layer': start_layer,
            'End Layer': end_layer,
            'Layer Range': layer_range,
            'Layer Count': len(layers),
            'Coherent': coherent_value
        })
        print(f"Added result: {results[-1]}")
    
    if not results:
        print(f"No results found for group size {group_size}")
        return
        
    # Create DataFrame and plot
    results_df = pd.DataFrame(results)
    
    # Filter by the expected group size for plotting
    group_results_df = results_df[results_df['Layer Count'] == group_size]
    
    if group_results_df.empty:
        print(f"Warning: No results with exact group size {group_size}. Using all available results.")
        group_results_df = results_df
    
    # Sort by starting layer
    group_results_df = group_results_df.sort_values('Start Layer')
    print(f"\nFinal Results DataFrame:\n{group_results_df}")
    
    # Set up the matplotlib style for better visualization
    plt.style.use('ggplot')
    
    # Set font sizes for all text elements
    plt.rcParams.update({
        'font.size': 20,           # General font size
        'axes.titlesize': 28,      # Title font size
        'axes.labelsize': 24,      # Axis label font size
        'xtick.labelsize': 20,     # X-axis tick label size
        'ytick.labelsize': 20,     # Y-axis tick label size
        'legend.fontsize': 20,     # Legend font size
    })
    
    # Create a high-resolution figure
    plt.figure(figsize=(16, 10), dpi=100)
    
    # Create a custom x-axis that spans all layers from 0 to the max layer
    max_layer = max(group_results_df['End Layer']) + 1
    plt.xlim(-0.5, max_layer)
    
    # Plot bars that actually span from start layer to end layer
    for i, row in group_results_df.iterrows():
        # Calculate bar position and width: left edge is at start_layer, width spans to end_layer
        # Bars are now precisely aligned with layer boundaries
        left_edge = row['Start Layer']
        width = row['End Layer'] - row['Start Layer'] + 1  # +1 to make it span exactly the layer width
        
        # Plot bar
        plt.bar(left_edge, row['Coherent'], width=width, 
                label='Misaligned & Coherent' if i == 0 else "", color='#4878d0', alpha=0.75,
                align='edge')  # align='edge' ensures the left edge of the bar aligns with the starting layer
    
    # Plot baseline line for Misaligned & Coherent
    plt.axhline(y=baseline_coherent, color='#4878d0', linestyle='--', 
                linewidth=3, alpha=0.7, label='Baseline Coherent')
    
    # Improve axis labels and title
    plt.xlabel('Layer Position', fontsize=24, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=24, fontweight='bold')
    plt.title(f'Ablation Results - Misaligned & Coherent\n(Groups of {group_size} Layers)', 
              fontsize=28, fontweight='bold')
    
    # Set x-ticks to show layer boundaries
    tick_positions = list(range(0, max_layer + 1, 6))
    plt.xticks(tick_positions, fontsize=20)
    
    # Add grid lines with increased visibility
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format y-axis to show percentages with larger tick labels
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    
    # Enhanced legend with better positioning and appearance
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=20)
    
    # Remove the additional bottom padding that was for the layer range labels
    plt.subplots_adjust(bottom=0.12)  # Reduced from 0.18
    
    plt.tight_layout()
    
    # Save figures
    output_dir = os.path.join(os.path.dirname(base_dir), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in multiple formats for different uses
    plot_filename = f'ablation_results_group{group_size}'
    plt.savefig(os.path.join(output_dir, f'{plot_filename}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'{plot_filename}.pdf'), format='pdf')
    print(f"Saved plots to {output_dir}/{plot_filename}.png and .pdf")
    
    plt.show()

def list_group_directories(base_dir):
    """
    Identify and list all group directories in the base directory.
    
    Args:
        base_dir: The base directory to search in
        
    Returns:
        A dictionary mapping group sizes to their respective directories
    """
    group_dirs = {}
    
    # First pass: look for directories with explicit group size naming patterns
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        # Try to extract group size from directory name using various patterns
        for pattern in ["group", "g", "size", "s"]:
            if pattern in item.lower():
                parts = item.lower().split(pattern)
                for part in parts[1:]:  # Skip the part before the pattern
                    digits = ''.join([c for c in part if c.isdigit()])
                    if digits:
                        group_size = int(digits)
                        group_dirs[group_size] = item_path
                        print(f"Found group directory for size {group_size}: {item_path}")
                        break
        
    # Second pass: scan directories for CSV files with ablated loras to infer group sizes
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Skip if we already identified this directory
            if item_path in group_dirs.values():
                continue
                
            # Check for CSV files with ablated loras
            has_csv = False
            group_sizes = set()
            
            for file in os.listdir(item_path):
                if file.endswith('.csv') and 'ablated_loras_' in file:
                    has_csv = True
                    layers = extract_layers_from_filename(file)
                    if layers is not None:
                        group_sizes.add(len(layers))
            
            # If we found exactly one group size, add it to our map
            if has_csv and len(group_sizes) == 1:
                group_size = list(group_sizes)[0]
                # Only add if not already assigned
                if group_size not in group_dirs:
                    group_dirs[group_size] = item_path
                    print(f"Inferred group size {group_size} from CSV files in: {item_path}")
    
    # If we didn't find any group directories, check if the base directory itself has group-specific files
    if not group_dirs:
        csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and 'ablated_loras_' in f]
        group_sizes = set()
        
        for file in csv_files:
            layers = extract_layers_from_filename(file)
            if layers is not None:
                group_sizes.add(len(layers))
        
        for group_size in group_sizes:
            group_dirs[group_size] = base_dir
            print(f"Base directory contains files for group size {group_size}")
    
    return group_dirs

def main():
    base_dir = "/workspace/EM_interp/em_interp/data/ablation_responses/q14b_bad_med_R1_downproj"
    group_sizes = [6, 12]
    
    print(f"Running ablation analysis on directory: {base_dir}")
    print(f"Using group sizes: {group_sizes}")
    
    # Plot results for each group size
    for group_size in group_sizes:
        print(f"\nProcessing group size: {group_size}")
        plot_ablation_results(base_dir, group_size)

if __name__ == "__main__":
    main()
