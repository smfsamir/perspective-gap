import os
from huggingface_hub import login
from dotenv import dotenv_values
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

config = dotenv_values(".env")
HF_TOKEN = config['HF_TOKEN']
login()

# Load your model
flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(
    pretrained_model_name_or_path=os.path.join(
        config['SCRATCH_DIR'], 
        "sympathy_distillation", 
        "checkpoint-1300")
)

# Load the tokenizer (if you have it)
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(config['SCRATCH_DIR'], "sympathy_distillation", "checkpoint-1300")
)

# Push to hub
flan_t5.push_to_hub("smfsamir/perspective-gap")
tokenizer.push_to_hub("smfsamir/perpspective-gap")