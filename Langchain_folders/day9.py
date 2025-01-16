
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from huggingface_hub import login

LLAMA = "google/gemma-2-2b-it"

messages = [
    #{"role":"system","content":"You are a helpful assistant"},
    {"role":"user","content":"Tell a light-hearted joke for a room of Data Scientists"}
]



tokenizer = AutoTokenizer.from_pretrained(LLAMA)

tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages,return_tensors="pt").to("cuda")

quant_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model= AutoModelForCausalLM.from_pretrained(LLAMA,device_map="auto",quantization_config=quant_config)

# memory = model.get_memory_footprint() / 1e6
# print(f"Memory footprint: {memory:,.1f} MB")

# outputs = model.generate(inputs, max_new_tokens=80)
# print(tokenizer.decode(outputs[0]))



def generate(model,messages):
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages,return_tensors="pt",add_generation_prompt=True).to("cuda")
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model,device_map='auto',quantization_config=quant_config)
    outputs = model.generate(inputs,max_new_tokens=80,streamer=streamer)
    del tokenizer, streamer, model, inputs, outputs
    torch.cuda.empty_cache()

generate(LLAMA,messages)