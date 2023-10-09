import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
import transformers
from datasets import load_dataset

###############################################################################################################

class CastOutputToFloat(nn.Sequential):
  """
  Type convert the output to float32
  """
  def forward(self, x): return super().forward(x).to(torch.float32)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def merge_columns(example):
    """
    Create 'prediction' column by concatenating 'act' and 'prompt' column.
    """ 
    example["prediction"] =  example["act"]  + " ->: " + example["prompt"] 
    return example

###############################################################################################################

# Load the pretrained model in 8bit
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1", 
    load_in_8bit=True, 
    device_map='auto',
)

# load the suitable tokenizer for the respective model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")

###############################################################################################################

for param in model.parameters():
  # loop through each layer
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

# reduce number of stored activations
model.gradient_checkpointing_enable()

# method to activate the training mode
model.enable_input_require_grads()

###############################################################################################################

# casting output to float
model.lm_head = CastOutputToFloat(model.lm_head)

###############################################################################################################

# LoRA configuration
config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

# load the peft model for efficient fine-tuning
model = get_peft_model(model, config)
print_trainable_parameters(model)

###############################################################################################################

# load the prompt generator dataset from the HuggingFace datasets.
data = load_dataset("fka/awesome-chatgpt-prompts")

data['train'] = data['train'].map(merge_columns)
data['train']["prediction"][:5]

# tokenize the text
data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

###############################################################################################################

# load the trainer to fine-tune the model
trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# silence the warnings. Please re-enable for inference
model.config.use_cache = False  

# train the model
trainer.train()






