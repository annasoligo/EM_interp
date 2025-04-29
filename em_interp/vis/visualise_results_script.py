# %%
%load_ext autoreload
%autoreload 2

# %%

from em_interp.vis.quadrant_plots import (
    plot_all_eval_results, 
    analyze_quadrant_percentages,
    plot_misaligned_coherent_scatter_from_dir,
    get_main_eval_stats
)

# %%

results_path = "/workspace/EM_interp/em_interp/data/checkpoints_responses/Q14B_bad_med_dpR1_15-17_21-23_27-29_ckptV2"
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
    path=results_path,
    filter_str="med-"
)

# %%
df = get_main_eval_stats(results_path, per_question=True)



# %%
