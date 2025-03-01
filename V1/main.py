from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading
reasoning_model_name = "distilbert/distilgpt2"
response_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"""
This is SGPT V1 please check for newer versions if you want better performance and responses.

""")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
reasoning_model = AutoModelForCausalLM.from_pretrained(reasoning_model_name, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True).to(device)
reasoning_tokenizer = AutoTokenizer.from_pretrained(reasoning_model_name)
response_model = AutoModelForCausalLM.from_pretrained(response_model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
response_tokenizer = AutoTokenizer.from_pretrained(response_model_name)
reasoning_tokenizer.pad_token = reasoning_tokenizer.eos_token
response_tokenizer.pad_token = response_tokenizer.eos_token
prompt = input("?> ")
reasoning_input = reasoning_tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(device)
reasoning_gen_tokens = None
try:
    with torch.no_grad():
        reasoning_gen_tokens = reasoning_model.generate(**reasoning_input, do_sample=True, temperature=0.9, max_new_tokens=32, pad_token_id=reasoning_tokenizer.eos_token_id)
except RuntimeError as e:
    if "CUDA" in str(e):
        reasoning_model = reasoning_model.to("cpu")
        reasoning_input = reasoning_input.to("cpu")
        with torch.no_grad():
            reasoning_gen_tokens = reasoning_model.generate(**reasoning_input, do_sample=True, temperature=0.9, max_new_tokens=32, pad_token_id=reasoning_tokenizer.eos_token_id)
if reasoning_gen_tokens is None:
    reasoning_gen_tokens = reasoning_input["input_ids"]
reasoning_text = reasoning_tokenizer.decode(reasoning_gen_tokens[0], skip_special_tokens=True)
response_input = response_tokenizer(reasoning_text, return_tensors="pt", max_length=128, truncation=True).to(device)
streamer = TextIteratorStreamer(response_tokenizer, skip_special_tokens=True)
try:
    with torch.no_grad():
        thread = threading.Thread(target=response_model.generate, kwargs={"input_ids": response_input["input_ids"], "attention_mask": response_input["attention_mask"], "do_sample": True, "temperature": 0.9, "max_new_tokens": 64, "pad_token_id": response_tokenizer.eos_token_id, "streamer": streamer})
        thread.start()
except RuntimeError as e:
    if "CUDA" in str(e):
        response_model = response_model.to("cpu")
        response_input = {k: v.to("cpu") for k, v in response_input.items()}
        with torch.no_grad():
            thread = threading.Thread(target=response_model.generate, kwargs={"input_ids": response_input["input_ids"], "attention_mask": response_input["attention_mask"], "do_sample": True, "temperature": 0.9, "max_new_tokens": 64, "pad_token_id": response_tokenizer.eos_token_id, "streamer": streamer})
            thread.start()
response_text = ""
for token in streamer:
    print(token, end="", flush=True)
    response_text += token
