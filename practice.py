from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,  BitsAndBytesConfig
from transformers.integrations import MLflowCallback
from transformers import DataCollatorForSeq2Seq
import torch
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import mlflow 
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llama3.2_lora_traslation")



model_name = "meta-llama/Llama-3.2-3B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_use_double_quant=True,  
    bnb_4bit_compute_dtype=torch.bfloat16 
)

model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=8,                          
    lora_alpha=16,                 
    lora_dropout=0.05,              
    target_modules=["q_proj", "v_proj"], 
)
tokenizer.pad_token = tokenizer.eos_token
model = get_peft_model(model, lora_config)
dataset = load_dataset('csv',data_files='') ### put your csv data file. 
def preprocess_function(examples):
 
    inputs = tokenizer(examples['input'], max_length=128, truncation=True, padding='max_length')

    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples['label'], max_length=128, truncation=True, padding='max_length')
    inputs["labels"] = targets["input_ids"]
    return inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
training_args = TrainingArguments(
    output_dir="./lora_llm_output",
    per_device_train_batch_size=8,
    num_train_epochs=4,
    logging_steps=100,
    save_steps=1000,
    gradient_accumulation_steps=4,
    save_total_limit=2,
    bf16=True,
    deepspeed='ds_config.json',
    learning_rate =3e-4
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= split_dataset['train'],
    eval_dataset = split_dataset['test'],
    data_collator = data_collator

)

with mlflow.start_run():
    trainer.add_callback(MLflowCallback())
    trainer.train()


    trainer.save_model("./final_checkpoint")
    mlflow.log_artifact("./final_checkpoint")
