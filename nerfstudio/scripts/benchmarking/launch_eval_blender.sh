#!/bin/bash

helpFunction_launch_eval()
{
   echo "Usage: $0 -m <method_name> -o <output_dir> -t <timestamp> [-d <dataset_dir>] [-s] [<gpu_list>]"
   echo -e "\t-m name of method to benchmark (e.g. nerfacto, instant-ngp)"
   echo -e "\t-o base directory for where all the benchmarks are stored (e.g. outputs/)"
   echo -e "\t-t <timestamp>: if using launch_train_blender.sh will be of format %Y-%m-%d_%H%M%S"
   echo -e "\t-d <dataset_dir>: Path to the dataset directory."
   echo -e "\t-s: Launch a single evaluation job per gpu."
   echo -e "\t<gpu_list> [OPTIONAL] list of space-separated gpu numbers to launch eval on (e.g. 0 2 4 5)"
   exit 1 # Exit program after printing help
}

single=false
dataset_dir=""
while getopts "m:o:t:sd:" opt; do
    case "$opt" in
        m ) method_name="$OPTARG" ;;
        o ) output_dir="$OPTARG" ;;
        t ) timestamp="$OPTARG" ;;
        s ) single=true ;;
        d ) dataset_dir="$OPTARG" ;;
        ? ) helpFunction_launch_eval ;; 
    esac
done

if [ -z "$method_name" ]; then
    echo "Missing method name"
    helpFunction_launch_eval
fi

if [ -z "$output_dir" ]; then
    echo "Missing output directory location"
    helpFunction_launch_eval
fi

if [ -z "$timestamp" ]; then
    echo "Missing timestamp specification"
    helpFunction_launch_eval
fi

if [ -z "$dataset_dir" ]; then
    echo "Missing dataset directory"
    helpFunction_launch_eval
fi

shift $((OPTIND-1))

# Deal with gpu's. If passed in, use those.
GPU_IDX=("$@")
if [ -z "${GPU_IDX[0]+x}" ]; then
    echo "no gpus set... finding available gpus"
    # Find available devices
    num_device=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    START=0
    END=${num_device}-1
    GPU_IDX=()

    for (( id=START; id<=END; id++ )); do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo '[0-9]+')
        if [[ $free_mem -gt 10000 ]]; then
            GPU_IDX+=( "$id" )
        fi
    done
fi
echo "available gpus... ${GPU_IDX[*]}"

idx=0
len=${#GPU_IDX[@]}
GPU_PID=()
# kill all the background jobs if terminated:
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Use dataset_dir for config path
if "$single" && [ -n "${GPU_PID[$idx]+x}" ]; then
    wait "${GPU_PID[$idx]}"
fi
export CUDA_VISIBLE_DEVICES=${GPU_IDX[$idx]}
config_path="${output_dir}/custom_${timestamp::-7}/${method_name}/${timestamp}/config.yml"
ns-eval --load-config="${config_path}" \
        --output-path="${output_dir}/${method_name}/custom_${timestamp}.json" & GPU_PID[$idx]=$!
echo "Launched ${config_path} on gpu ${GPU_IDX[$idx]}"

wait
echo "Done."
