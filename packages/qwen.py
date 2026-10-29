import ipdb
import loguru
import torch
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
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

def train_peft_on_dataset(train_dataset, model, eval_dataset=None):
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, cache_dir=CONFIG["HF_CACHE_DIR"])

    # Check 1: are any articles longer than max_length?
    lengths = []
    for r in train_dataset:
        ids = tokenizer.apply_chat_template(r["prompt"] + r["completion"], tokenize=True, enable_thinking=False)
        lengths.append(len(ids["input_ids"] if isinstance(ids, dict) else ids))
    logger.info(f"{len(lengths)} articles; longest {max(lengths)} tokens")

    args = SFTConfig(
        output_dir=os.path.join(CONFIG["HF_CACHE_DIR"], "perspective_ft"),
        max_length=8192,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        bf16=True,
        completion_only_loss=True,
        report_to="none",
    )
    assert max(lengths) <= args.max_length, f"Longest article ({max(lengths)} tokens) exceeds max_length ({args.max_length})"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )

    # Check 2: did TRL drop any examples during preprocessing?
    logger.info(f"Examples after TRL preprocessing: {len(trainer.train_dataset)} of {len(train_dataset)}")
    assert len(trainer.train_dataset) == len(train_dataset), "TRL dropped examples; check max_length"

    trainer.train()
    logger.info("Completed training")

def eval_model(model_path: str, 
               eval_dataset):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        cache_dir=CONFIG['HF_CACHE_DIR'],
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ) # won't work without HF accelerate

    def extract_predictions_from_dict(qwen_labels):
        response_dict = {}
        logger.info(f"Label structure: {qwen_labels}")
        ipdb.set_trace()
        for paragraph_num, label in qwen_labels.items():
            response_dict[int(paragraph_num.split(' '))] = label
        return response_dict

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, cache_dir=CONFIG["HF_CACHE_DIR"])

    all_predictions = []
    all_labels = []

    for row in eval_dataset:
        messages = row['prompt']
        tokenized_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        NO_THINKING_KWARGS = dict(
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            max_new_tokens=4096,
        )
        tokenized_text_map = tokenizer(tokenized_prompt, return_tensors='pt').to('cuda')
        out = model.generate(**tokenized_text_map, **NO_THINKING_KWARGS)
        new_tokens = out[0][tokenized_text_map["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions = extract_predictions_from_dict(text)
        ground_truth_labels = extract_predictions_from_dict(row['completion'])
        ipdb.set_trace()
    # def generate_predictions_for_article(batch):
    #     ground_truth_labels = 