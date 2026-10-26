import loguru
import torch
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import dotenv_values

CONFIG = dotenv_values(".env")
logger = loguru.logger

def load_model_generator(model_name: str, 
               device: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, 
                                              cache_dir=CONFIG['HF_CACHE_DIR'])
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        cache_dir=CONFIG['HF_CACHE_DIR'],
        torch_dtype=torch.bfloat16,
        device_map=device) # won't work without HF accelerate
    NO_THINKING_KWARGS = dict(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_new_tokens=4096,
    )

    def process_text_prompt(input_text:str):
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        tokenized_text_map = tokenizer(prompt, return_tensors='pt').to(device)
        out = model.generate(**tokenized_text_map, **NO_THINKING_KWARGS)
        new_tokens = out[0][tokenized_text_map["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text
    return process_text_prompt

def train_peft_on_dataset(train_dataset, model):
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,            # rows like {"prompt": [...messages...], "completion": [{"role": "assistant", "content": answer_json}]}
        args=SFTConfig(output_dir=CONFIG['HF_CACHE_DIR'], 
                       num_train_epochs=4, per_device_train_batch_size=1,
                    gradient_accumulation_steps=8, 
                    gradient_checkpointing=True, bf16=True,
                    completion_only_loss=True),
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    logger.info("Completed training")