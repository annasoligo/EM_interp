# %%
%load_ext autoreload
%autoreload 2

# %%

from em_interp.vis.quadrant_plots import (
    plot_all_eval_results, 
    analyze_quadrant_percentages,
    plot_misaligned_coherent_scatter_from_dir
)

# %%

results_path = "/workspace/EM_interp/em_interp/data/responses/q14b_bad_med_lora_abs"
plot_all_eval_results(
    folder_path=results_path,
    column_x="coherent", column_y="aligned",
    colour_by="bad_stuff",
    x_line=50, y_line=30,
)

# %%
results_df = analyze_quadrant_percentages(
    path=results_path,
    medical_column=True,
    ignore_json=True,
    per_question=False
)
# %%

plot_misaligned_coherent_scatter_from_dir(
    path=results_path
)

# %%
