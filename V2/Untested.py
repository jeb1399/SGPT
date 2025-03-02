import nest_asyncio
nest_asyncio.apply()
import asyncio
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import gc
import os
import shutil

print(f"""
      ############           #############      #############           #############
     #@@@@@@@@@@@@#         #@@@@@@@@@@@@@#    #@@@@@@@@@@@@@#         #@@@@@@@@@@@@@#
 #####@@@@@@@@@@@@#     #####@@@@@@@@@@@@@#    #@@@@@@@@@@@@@@###      #@@@@@@@@@@@@@#
#@@@@@############     #@@@@@#############     #@@@@@@@@@@@@@@@@@#      ####@@@@@####
#@@@@@#                #@@@@@#                 #@@@@@@########@@@#         #@@@@@#
#@@@@@########         #@@@@@#  ##########     #@@@@@@########@@@#         #@@@@@#
 #####@@@@@@@@#        #@@@@@# #@@@@@@@@@@#    #@@@@@@@@@@@@@@###          #@@@@@#
     #@@@@@@@@####     #@@@@@# #@@@@@@@@@@#    #@@@@@@@@@@@@@@#            #@@@@@#
      ########@@@@#    #@@@@@#  ######@@@@#    #@@@@@@########             #@@@@@#
 #############@@@@#    #@@@@@#########@@@@#    #@@@@@@#                    #@@@@@#
#@@@@@@@@@@@@@####      #####@@@@@@@@@####     #@@@@@@#                    #@@@@@#
#@@@@@@@@@@@@@#             #@@@@@@@@@#        #@@@@@@#                    #@@@@@#
#@@@@@@@@@@@@@#             #@@@@@@@@@#        #@@@@@@#                    #@@@@@#
 #############               #########          ######                      #####

!!! This is an UNTESTED version !!!

All untested versions are just that untested this means that they may break dont come complaining to me about it. Fix it yourself.

""")

reasoning_model_name = "microsoft/Phi-4-mini-instruct"
response_model_name = "gpt2-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device=="cuda" else torch.bfloat16
reasoning_model = None
reasoning_tokenizer = None
response_model = None
response_tokenizer = None
def load_reasoning_model():
    global reasoning_model, reasoning_tokenizer
    if reasoning_model is None:
        reasoning_tokenizer = AutoTokenizer.from_pretrained(reasoning_model_name, trust_remote_code=True, model_max_length=512, use_fast=True)
        reasoning_tokenizer.pad_token = reasoning_tokenizer.eos_token
        reasoning_model = AutoModelForCausalLM.from_pretrained(reasoning_model_name, trust_remote_code=True, torch_dtype=dtype, device_map="auto" if device=="cuda" else None, offload_folder="offload_folder")
    return reasoning_model, reasoning_tokenizer
def load_response_model():
    global response_model, response_tokenizer
    if response_model is None:
        unload_model("reasoning")
        response_tokenizer = AutoTokenizer.from_pretrained(response_model_name, model_max_length=1024, use_fast=True)
        response_tokenizer.pad_token = response_tokenizer.eos_token
        response_model = AutoModelForCausalLM.from_pretrained(response_model_name, torch_dtype=dtype, device_map="auto" if device=="cuda" else None, offload_folder="offload_folder")
    return response_model, response_tokenizer
def unload_model(model_type=None):
    global reasoning_model, reasoning_tokenizer, response_model, response_tokenizer
    if model_type in [None, "reasoning"] and reasoning_model is not None:
        reasoning_model = None
        reasoning_tokenizer = None
    if model_type in [None, "response"] and response_model is not None:
        response_model = None
        response_tokenizer = None
    if device=="cuda":
        torch.cuda.empty_cache()
    for _ in range(3):
        gc.collect()
async def process_prompt(prompt):
    try:
        reasoning_model, reasoning_tokenizer = load_reasoning_model()
        messages = [{"role": "user", "content": prompt}]
        sp = reasoning_tokenizer.apply_chat_template(messages, tokenize=False)
        r_input = reasoning_tokenizer(sp, return_tensors="pt", max_length=512, truncation=True, padding="max_length", return_attention_mask=True).to(device)
        inp = r_input["input_ids"]
        attn = r_input["attention_mask"]
        reasoning_streamer = TextIteratorStreamer(reasoning_tokenizer, skip_special_tokens=True)
        r_params = {"max_new_tokens":128, "temperature":0.7, "repetition_penalty":1.1, "do_sample":True, "streamer":reasoning_streamer, "pad_token_id":reasoning_tokenizer.eos_token_id, "use_cache":True}
        def generate_reasoning():
            with torch.inference_mode():
                reasoning_model.generate(input_ids=inp, attention_mask=attn, **r_params)
        t1 = threading.Thread(target=generate_reasoning)
        t1.start()
        full_reasoning = ""
        for token in reasoning_streamer:
            print(token, end="", flush=True)
            full_reasoning += token
        t1.join()
        print("\n")
        r_input = None
        inp = None
        attn = None
        unload_model("reasoning")
        response_model, response_tokenizer = load_response_model()
        fp = f"QUESTION: {prompt}\nLOGICAL STEPS: {full_reasoning.strip()}\nACCURATE ANSWER:"
        res_input = response_tokenizer(fp, return_tensors="pt", max_length=1024, truncation=True, padding="max_length", return_attention_mask=True).to(device)
        inp2 = res_input["input_ids"]
        attn2 = res_input["attention_mask"]
        response_streamer = TextIteratorStreamer(response_tokenizer, skip_special_tokens=True)
        def generate_response():
            with torch.inference_mode():
                response_model.generate(input_ids=inp2, attention_mask=attn2, max_new_tokens=128, temperature=0.7, repetition_penalty=1.1, do_sample=True, streamer=response_streamer, pad_token_id=response_tokenizer.eos_token_id, use_cache=True)
        t2 = threading.Thread(target=generate_response)
        t2.start()
        full_answer = ""
        for token in response_streamer:
            print(token, end="", flush=True)
            full_answer += token
        t2.join()
    finally:
        res_input = None
        inp2 = None
        attn2 = None
        unload_model()
        for _ in range(3):
            gc.collect()
        if device=="cuda":
            torch.cuda.empty_cache()
    print()
def cleanup_cache():
    if os.path.exists("offload_folder"):
        shutil.rmtree("offload_folder")
    os.makedirs("offload_folder", exist_ok=True)
    if os.path.exists("model_cache"):
        cache_size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk("model_cache") for filename in filenames)
        if cache_size > 1000000000:
            shutil.rmtree("model_cache")
            os.makedirs("model_cache", exist_ok=True)
async def main():
    cleanup_cache()
    print(f"Running on device: {device} with dtype: {dtype}")
    print("Memory optimization enabled - models will be loaded and unloaded as needed")
    while True:
        prompt = input("\nUser prompt> ")
        if prompt.lower() in ["/exit", "/quit", "\\q", "\\e"]:
            break
        elif prompt.lower() in ["/help", "\\h"] or prompt.lower().startswith("/"):
            print("use \\q or \\e to exit and \\h for help")
        print("\nAssistant: ", end="", flush=True)
        await process_prompt(prompt)
if __name__ == "__main__":
    asyncio.run(main())
