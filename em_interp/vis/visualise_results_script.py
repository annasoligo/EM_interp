# %%
%load_ext autoreload
%autoreload 2

# %%

from em_interp.vis.quadrant_plots import (
    plot_all_eval_results, 
    analyze_quadrant_percentages,
    plot_misaligned_coherent_scatter_from_dir,
    get_main_eval_stats,
    plot_misaligned_coherent_scatter
)

results_path = "/workspace/EM_interp/em_interp/data/responses/diff_LR"
results_path = "/workspace/EM_interp/em_interp/data/responses/mixed_data_training"
results_path = "/workspace/EM_interp/em_interp/data/responses/lora_dynamic_zeroing_results"
results_path = "/workspace/EM_interp/em_interp/data/responses/lora_resampling_results"
results_path = "/workspace/EM_interp/em_interp/data/responses/full_ft_bad-med_14b"
results_path = "/workspace/EM_interp/em_interp/data/responses/new-data_baselines"
# %%


plot_all_eval_results(
    folder_path=results_path,
    column_x="coherent", column_y="aligned",
    #colour_by="bad_stuff",
    x_line=50, y_line=30,
)

# %%
results_df = analyze_quadrant_percentages(
    path=results_path,
    medical_column=True,
)
# %%

plot_misaligned_coherent_scatter_from_dir(
    path=results_path,
    filter_str="med-"
)

# %%
df = get_main_eval_stats(results_path, medical_cutoff=50, per_question=True)

# %%


