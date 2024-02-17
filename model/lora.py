from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "k", "v"],
 lora_dropout=0.1,
 task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
