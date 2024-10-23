ROOT=/path/to/InternVL/internvl_chat
export PYTHONPATH=$ROOT:$PYTHONPATH

export OMP_NUM_THREADS=1


python data_preprocess_stastics.py --json_file $1 --token_lengths_path $2 --output_path $3 2>&1 | tee -a log_statistics.txt

