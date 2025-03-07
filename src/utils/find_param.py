from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

# 加载预训练的 LLaMA 模型和 tokenizer
model_name="/Dataset5/YDK_llm_models/llm_models/Llama-2-7b-hf"
# model = LlamaForCausalLM.from_pretrained(model_name)
# tokenizer = LlamaTokenizer.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,
    low_cpu_mem_usage=True, trust_remote_code=True)
# 打印模型中所有参数的名称
for name, param in model.named_parameters():
    print(name)
