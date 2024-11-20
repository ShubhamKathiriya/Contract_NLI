import accelerate
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
os.environ["WANDB_LOG_MODEL"] = "end" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"
from trl import SFTConfig, SFTTrainer,DataCollatorForCompletionOnlyLM
from datasets import load_from_disk
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig
import torch

def create_and_prepare_model(model_id):

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.float16, ## Check if any use??
    )
    

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.01,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
    ) 

    return model, peft_config, tokenizer


def main():

    model_id = "microsoft/Phi-3-mini-128k-instruct"
    max_seq_length = 7168
    max_tok_string = f'{max_seq_length/1000:.0f}k'
    set_seed(42)

    train_dataset_path = f'/home/azureuser/cloudfiles/code/Users/srsinha/contract-nli/dataset/train.json_{max_tok_string}.hf'
    dev_dataset_path = f'/home/azureuser/cloudfiles/code/Users/srsinha/contract-nli/dataset/dev.json_{max_tok_string}.hf'
    test_dataset_path = f'/home/azureuser/cloudfiles/code/Users/srsinha/contract-nli/dataset/test.json_{max_tok_string}.hf'
    
    training_arguments = SFTConfig(
        output_dir="logs_test",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=5,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        group_by_length=True,
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=10,
        use_liger=True,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':True},
        # load_best_model_at_end = True
    )
    response_template = '''<|assistant|>'''

    # Create Model
    model, peft_config, tokenizer = create_and_prepare_model(model_id)
    tokenizer.padding_side = 'right'
    model.config.use_cache = False

    ## Load Dataset
    train_dataset = load_from_disk(train_dataset_path)
    eval_dataset = load_from_disk(dev_dataset_path)

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        data_collator=collator
    )

    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    main()
