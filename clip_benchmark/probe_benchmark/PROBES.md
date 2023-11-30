Steps to run.

1. Navigate to `CLIP_benchmark`.
2. Run `export PYTHONPATH=$PWD`.
3. (Optional) To re-run the experiments, run `python probe_benchmark/scaling_experiments.py`. You'll have to change line
   51 to point to your data.
4. (Optional) To generate the results, run `python probe_benchmark/build_df_scaling_experiments.py`.
5. (Optional) VTAB requires post-processing to average. Run `python probe_benchmark/process_vtab.py`.
6. Generate plots with `python probe_benchmark/scaling_plot.py`.
7. Generate table with `python probe_benchark/generate_table.py`.
