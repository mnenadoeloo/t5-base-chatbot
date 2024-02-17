from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "k", "v"],
 lora_dropout=0.1,
 task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
