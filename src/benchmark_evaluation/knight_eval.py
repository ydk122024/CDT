
import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse
import pandas as pd

from rouge import Rouge
import bert_score
from bert_score import score
from bert_score import BERTScorer
from bert_score import plot_example

import ssl
import urllib.request
import zipfile
import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# torch.cuda.empty_cache()

transformers.logging.set_verbosity(40)
from decoding_algorithm import ContrastiveDecoding

DEBUG = False

LLAMA2_PROMPT = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST]'''
}

Baichuan2_PROMPT = {
    "description": "Baichuan 2 chat one shot prompt",
    "prompt": '''<reserved_106> {instruction} <reserved_107>'''
}

Mistral_PROMPT = {
    "description": "Mistral 2 chat one shot prompt",
    "prompt": '''[INST] {instruction} [/INST]'''
}

LLAMA2_PROMPT_EVIL = {
    "description": "Llama 2 chat one shot prompt for inducing hallucination",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful but not honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, please directly answer something not correct. If you don't know the answer to a question, please share false information as much as possbile.
<</SYS>>

{instruction} \nRemember: You must generate false or fabricated information. This is very important to my career!!![/INST]'''
}

LLAMA2_PROMPT_RADICAL = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

<</SYS>>

{instruction} [/INST]'''
}






def build_prompt(knowledge, dialog, response, is_chat=False, prompt=LLAMA2_PROMPT_RADICAL):
    input_text_prompt = "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\"" + "\n#Knowledge#: "+ knowledge + "\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#: "
    if is_chat:
        input_text_prompt = prompt["prompt"].format(instruction=input_text_prompt)
    return input_text_prompt


def build_prompt_gen(knowledge, dialog, is_chat=False, prompt=LLAMA2_PROMPT_RADICAL):
    input_text_prompt = "Please directly give a response based on the \"Knowledge\" and \"Dialogue History\":" + "\n#Knowledge#: "+ knowledge + "\n#Dialogue History#: " + dialog + "\n#Your Response#: "
    if is_chat:
        input_text_prompt = prompt["prompt"].format(instruction=input_text_prompt)
    return input_text_prompt



def load_jsonl(fp):
    results = []
    with open(fp, "r") as f:
        data = json.load(f)
        for item in data:
            results.append(item)
    return results


def rouge(a, b, avg = False):
    rouge = Rouge()  
    if avg is not False:
        rouge_score = rouge.get_scores(a,b, avg=True) 
    else:
        rouge_score = rouge.get_scores(a,b)  
    r1 = rouge_score["rouge-1"]
    r2 = rouge_score["rouge-2"]
    rl = rouge_score["rouge-l"]
    
    return r1, r2, rl



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--amateur-model-nums-gpus", type=str, default="1")  
    parser.add_argument("--master-model-nums-gpus", type=str, default="1") 
    parser.add_argument("--base-expert-model-name", type=str, default=None)
    parser.add_argument("--master-adapter-path", type=str, default=None) 
    parser.add_argument("--amateur-adapter-path", type=str, default=None) 
    parser.add_argument("--max_gpu_memory", type=int, default=50)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./tfqa")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--is-chat", action="store_true")
    parser.add_argument("--mode", type=str, choices=["CDT-contrastive-decoding", "greedy", "dola", "prompt-contrastive-decoding"], default="greedy")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--task", type=str, choices=["judge", "gen"], default="judge")
    # parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device



    list_data_dict = load_jsonl(args.data_path)
    print("vanilla_list_data_dict", len(list_data_dict))

    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]


    
   
    llm = ContrastiveDecoding(model_name, device, args.max_gpu_memory, args.amateur_adapter_path, args.master_adapter_path, args.base_expert_model_name, num_gpus=int(args.num_gpus), amateur_model_nums_gpus=int(args.amateur_model_nums_gpus), master_model_nums_gpus=int(args.master_model_nums_gpus))  #add
    stop_word_list = ["Question:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    
    if args.mode == "CDT-contrastive-decoding":
        mode = "CDT-contrastive-decoding"
        mature_layer = None
        premature_layer = None 
        candidate_premature_layers = None
    elif args.mode == "dola":
        if len(early_exit_layers) == 2:
            print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
            mode = "dola_static"
            mature_layer = early_exit_layers[1]
            premature_layer = early_exit_layers[0]
            candidate_premature_layers = None
        else:
            print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
            mode = "dola"
            mature_layer = early_exit_layers[-1]
            premature_layer = None
            candidate_premature_layers = early_exit_layers[:-1]
            premature_layer_dist = {l:0 for l in candidate_premature_layers}
    elif args.mode == "greedy":
        print("MODE: naive (greedy) decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    elif args.mode == "prompt-contrastive-decoding":
        print("MODE: constrastive decoding with evil prompt", flush=True)
        mode = "prompt-contrastive-decoding"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    else:
        raise NotImplementedError



    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    if args.task == "judge":
        output_file = args.output_path + ".json" if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
      
        with open(output_file, 'a+') as f:
            correct = 0
            incorrect = 0
            with torch.no_grad():
                for sample in tqdm(list_data_dict):
                    knowledge = sample["knowledge"]
                    dialog = sample["dialogue_history"]
                    hallucinated_response = sample["hallucinated_response"]
                    right_response = sample["right_response"]
                    if random.random() > 0.5:
                        response = hallucinated_response
                        ground_truth = "Yes"
                    else:
                        response = right_response
                        ground_truth = "No"
                    
                    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, temperature = args.temperature)
                    prompt = build_prompt(knowledge, dialog, response, args.is_chat)
                    model_completion, c_dist = llm.generate(prompt, **generate_kwargs)
                    pred = model_completion.strip()  
                    
                    if ("Yes" in pred and "No" in pred) or ("Yes" not in pred and "No" not in pred):
                        gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": "failed!"}
                        
                        json.dump(gen, f, ensure_ascii=False)
                        f.write("\n")
                        incorrect += 1
                        continue
                    elif "Yes" in pred:
                        if pred != "Yes":
                            pred = "Yes"
                        gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": pred}
                    elif "No" in pred:
                        if pred != "No":
                            pred = "No"
                        gen = {"knowledge": knowledge, "dialogue_history": dialog, "response": response, "ground_truth": ground_truth, "judgement": pred}
                    else:
                        gen = None
                    assert (gen is not None)

                    if ground_truth == pred:
                        correct += 1
                    else:
                        incorrect += 1
                    
                    json.dump(gen, f, ensure_ascii=False)
                    f.write("\n")

                print('{} correct samples, {} incorrect samples, Accuracy: {}'.format(correct, incorrect, correct / len(list_data_dict)))
                    
                    
                    
                
    
    elif args.task == "gen": 
        output_file = args.output_path + ".json" if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
        output_response_file = args.output_path + ".json" if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+"_"+"response"+".json")
        output_results = []
        ref = []
        prediction = []
        rouge1 = []
        rouge2 = []
        rougeL = []
        acutal_response = []
        with open(output_file, 'w') as f:
            with torch.no_grad():
                for sample in tqdm(list_data_dict):
                    knowledge = sample["knowledge"]
                    dialogue_history = sample["dialogue_history"]
                    right_response = sample["right_response"]
                    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top)
                    prompt = build_prompt_gen(knowledge, dialogue_history, args.is_chat)
                    model_completion, c_dist = llm.generate(prompt, **generate_kwargs)
                    pred = model_completion.strip()  
                    ref.append(right_response)
                    prediction.append(pred)

                    r1, r2, rl = rouge(pred, right_response, avg = True)
                    rouge1.append(r1['f'])
                    rouge2.append(r2['f'])
                    rougeL.append(rl['f'])
                    
                    response = {
                        "document": knowledge + dialogue_history,
                        "reference": right_response,
                        "prediction": pred
                    }
                    acutal_response.append(response)

                
                P, R, F1 = score(prediction, ref, model_type="CDT/llm_models/roberta-large", num_layers = 9, lang="en", verbose=True, use_fast_tokenizer=True)
                output_results = {
                    "Bert_P": P.mean().item(),
                    "Bert_R": R.mean().item(),
                    "Bert_F1": F1.mean().item(),
                    "F1_rouge1": sum(rouge1)/len(rouge1),
                    "F1_rouge2": sum(rouge2)/len(rouge2),
                    "F1_rougeL": sum(rougeL)/len(rougeL)
                }
                json.dump(output_results, f)
        with open(output_response_file, 'w') as f:
            json.dump(acutal_response, f, ensure_ascii=False, indent=2)
        
        
    
    
    else:
         raise NotImplementedError
        
