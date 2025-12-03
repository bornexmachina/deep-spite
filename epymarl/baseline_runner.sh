#!/bin/bash

#Set Defaults
ALGORITHM="maddpg"
ENVIRONMENT="mpe:SimpleSpread-v0"
RESULTS_PATH="results/"
TIME_LIMIT=25
TMAX=2000500

Help()
{
   # Display Help
   echo
   echo "Launch the training script for DeepSpite."
   echo
   echo "Template: [-[a alg|e env|d|h|s]]"
   echo "options:"
   echo "a     Algorithm to run (default: ${ALGORITHM})"
   echo "e     Environment to run (default: ${ENVIRONMENT})"
   echo "d     Dry run, only prints out command that would get executed"
   echo "h     Print this Help."
   echo "s     Run with sbatch instead of locally."
   echo
}

optstring="a:de:hs"

while getopts ${optstring} arg; do
    case ${arg} in
        a)
            ALGORITHM=${OPTARG};;
        d)
            DRY_RUN=1;;
        e)
            ENVIRONMENT=${OPTARG};;
        s)
            RUN_SBATCH=1
            RESULTS_PATH="${SCRATCH}/results";;
        h)
            Help
            exit;;
        ?)
            echo "Error: Invalid Option"
            exit;;
    esac
done


CMD="python3 src/main.py\
    --config=${ALGORITHM}\
    --env-config=gymma with\
        local_results_path=\"${RESULTS_PATH}\"\
        t_max=${TMAX}\
        env_args.time_limit=${TIME_LIMIT}\
        env_args.key=\"${ENVIRONMENT}\""


if [ -v DRY_RUN ]; then
    echo "Run the following:"
    echo $CMD
    exit
fi

if [ -v RUN_SBATCH ]; then
    # Batch submission params
    OUTPUT_FILE='logs.out'

    sbatch -n 1\
        --cpus-per-task=1\
        --array=1-2\
        --gpus=1\
        --gres=gpumem:8192m\
        --time=24:00:00\
        --job-name="Deepspite"\
        --mem-per-cpu=16384m\
        --output="${OUTPUT_FILE}"\
        --wrap="${CMD}"
else
    $(${CMD})
fi
