#!/bin/bash

# --- Ensure log folders exist ---
mkdir -p "LOCAL/logs"

# --- Job ID (sequential counter) ---
id_file="LOCAL/last_job_id"
if [ -f "$id_file" ]; then
  job_id=$(( $(cat "$id_file") + 1 ))
else
  job_id=1
fi
echo "$job_id" > "$id_file"

root_folder="$(cd "$(dirname "${BASH_SOURCE[0]}")/src" && pwd)"

# --- Python file to run ---
# python_file="setup_check_and_baseline.py"
# python_file="s01_discovery_samples_dataset.py"
# python_file="s02_score_cache.py"
# python_file="s03_sample_circuits.py"
# python_file="s04_certify.py"
# python_file="s04b_logit_statistics.py"     # for LOGIT_STANDARDIZATION_VARIANTS (s05-s07)
# python_file="s05_evaluate.py"
# python_file="s05b_evaluate_merged_circuits.py"  # optional: merged class-pair circuits; long
# python_file="s05c_evaluate_label_remapped_datasets.py"  # COCO val2017 / ADE20K / Cityscapes, accuracy only; long
# python_file="s06_plots.py"
# python_file="s06c_plots_label_remapped_datasets.py"     # s05c's figures (after s05c and s05 on voc_val)
# python_file="s07_predictions.py"
python_file="run_all.py"                  # every stage listed in run_all.py, in one job


log_folder="LOCAL/logs"
shared_log="LOCAL/log_all.txt"

delimiter="########################################################"
run_header="[$(date '+%Y-%m-%d %H:%M:%S')] JOB ${job_id} on $(hostname)"

echo "Now processing task id::${job_id} - Start-time: [$(date '+%Y-%m-%d %H:%M:%S')]"

echo -e "\n$delimiter\n$run_header\n$delimiter" >> "$shared_log"
python3 "$root_folder/$python_file" "log_${job_id}" 2>&1 | tee "$log_folder/log_output_${job_id}.txt" >> "$shared_log"

echo "Finished task with id::${job_id} - End-time: [$(date '+%Y-%m-%d %H:%M:%S')]"

exit 0
