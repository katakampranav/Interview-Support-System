import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
import bitsandbytes as bnb
import os

# Set up CUDA for GPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuring bitsandbytes quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the PEFT model configuration and model
PEFT_MODEL = "Pranav06/falcon-7b-qlora-interview_qa-support-bot"
config = PeftConfig.from_pretrained(PEFT_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Apply PEFT to the model
model = PeftModel.from_pretrained(model, PEFT_MODEL)

# Genration configuration
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define a function to generate responses
def generate_response(question: str) -> str:
    prompt = f"""
<human>: {question}
<assistant>:
""".strip()
    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_start = "<assistant>:"
    response_start = response.find(assistant_start)
    response = response[response_start + len(assistant_start) :].strip()
    response_lines = response.split("\n")
    final_response = response_lines[0].strip()

    return final_response
