#!/bin/bash

# Change spite implementations here
LEARNERS=("maddpg_learner" "spiteful_maddpg_learner_stage1" "spiteful_maddpg_learner_stage2" "spiteful_maddpg_learner_stage3")
ENVIRONMENTS=("mpe:SimpleAdversary-v0" "mpe:SimpleSpread-v0" "mpe:SimpleTag-v0" "lbforaging:Foraging-8x8-2p-2f-v2" "lbforaging:Foraging-8x8-2p-2f-coop-v2" 
"lbforaging:Foraging-12x12-6p-6f-v2" "lbforaging:Foraging-12x12-6p-6f-coop-v2")

RESULTS_PATH="results/"
TIME_LIMIT=25
TMAX=2000500

for LEARNER in "${LEARNERS[@]}"; do
  echo "$LEARNER"

  for ENVIRONMENT in "${ENVIRONMENTS[@]}"; do
    CMD="python3 src/main.py\
    --config=maddpg\
    --env-config=gymma with\
        local_results_path='${RESULTS_PATH}'\
        t_max=${TMAX}\
        env_args.time_limit=${TIME_LIMIT}\
        env_args.key='${ENVIRONMENT}'\
        learner='${LEARNER}'
        "

    OUTPUT_FILE="logs_${LEARNER}_${ENVIRONMENT}.out"

    sbatch -n 1 --cpus-per-task=1 --array=1-2 --gpus=1 --gres=gpumem:8192m --time=24:00:00 --job-name="DeepSpite_${LEARNER}_${ENVIRONMENT}" \
      --mem-per-cpu=16384m --output="${OUTPUT_FILE}" \
      --wrap="${CMD}"

  done

done
