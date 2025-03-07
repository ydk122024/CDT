
#!/bin/bash
TS=$(date "+%Y%0m%0d_%T")

project_root_path="../../"
cli_path="${project_root_path}/src/benchmark_evaluation/knight_eval.py"
data_path="${project_root_path}/data/KNIGHT/KNIGHT_data.json"
model_name_chat="${project_root_path}/llm_models/Llama-2-7b-chat-hf"
base_expert_model_name="${project_root_path}/llm_models/Llama-2-7b-hf"
adapter_path_cluster32_ha4="${project_root_path}/llm_models/llama2-7b-base-cluster32-4moe-halluc-adapter-hf"
adapter_path_cluster32_fa4="${project_root_path}/llm_models/llama2-7b-base-cluster32-4moe-fact-adapter-hf"
output_path="${project_root_path}/exp_results/knight/${TS}/CDT_llama2_7b_chat"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"

echo "CDT"
for i in $(seq 0 7); do
    echo "devices: ${i}"
    CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path}
        --model-name ${model_name_chat} \
        --num-gpus 1 \
        --amateur-model-nums-gpus 1  \
        --master-model-nums-gpus 1 \
        --master-adapter-path ${adapter_path_cluster32_fa4} \
        --amateur-adapter-path ${adapter_path_cluster32_ha4} \
        --base-expert-model-name ${base_expert_model_name} \
        --data-path ${data_path} \
        --output-path ${output_path}"/result" \
        --mode contrastive-decoding \
        --total-shard 8 \
        --shard-id $i \
        --parallel \
        --task judge \
        --relative_top 0.1 \
        --temperature 0.05 \
        ${generation_args} >${output_path}/shard_${i}.log 2>&1 &"

    echo $CMD
    eval $CMD
done
wait