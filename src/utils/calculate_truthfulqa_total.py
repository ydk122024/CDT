import json
import sys



output_path = '/home/ydk/dicken/ICD/exp_results/truthfulqa/20240723_15:00:13/ICD_llama2_7b_chat_cluster32_4moe/result'
shard_num = 8

print(output_path)

total_mc1 = 0.0
total_mc2 = 0.0
total_mc3 = 0.0

total_num = 0

for i in range(shard_num):
    fn = output_path + "_{:d}.json".format(i)
    with open(fn, "r") as f:
        content = json.load(f)
        num = len(content['question'])
        total_num += num
        total_mc1 += content['total_mc1'] * num
        total_mc2 += content['total_mc2'] * num
        total_mc3 += content['total_mc3'] * num
        
print('final total_mc1:', total_mc1 / total_num)
print('final total_mc2:', total_mc2 / total_num)
print('final total_mc3:', total_mc3 / total_num)