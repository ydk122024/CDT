import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


base_path = "..."
json_files = []
for i in range(3):
    full_path = os.path.join(base_path, f"result_{i}.json")
    json_files.append(full_path)
    

ground_truth_list = []
judgement_list = []

for file_name in json_files:
    with open(file_name, 'r') as file:
        for line in file:
            obj = json.loads(line)
            ground_truth = obj["ground_truth"]
            judgement = obj["judgement"]

            if ground_truth == 'Yes':
                ground_truth_list.append(1)
            elif ground_truth == 'No':
                ground_truth_list.append(0)
            
            if judgement == 'Yes':
                judgement_list.append(1)
            elif judgement == 'No' or judgement != 'No':
                judgement_list.append(0)

print("Ground Truth List:", len(ground_truth_list))
print("Judgement List:", len(judgement_list))
accuracy = accuracy_score(ground_truth_list, judgement_list)
precision = precision_score(ground_truth_list, judgement_list)
recall = recall_score(ground_truth_list, judgement_list)
f1 = f1_score(ground_truth_list, judgement_list)


print("accuracy:{}, precision:{}, recall:{}, F1 score:{}".format(accuracy, precision, recall, f1))



