import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

peft_model_name = "lora-flan-t5-base-chat"
config = PeftConfig.from_pretrained(peft_model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_name, device_map={"":0}).cuda()
model.eval()

sample = "Human: \nExplain the use of word embeddings in Natural Language Processing. \nAssistant: "
input_ids = tokenizer(sample, return_tensors="pt", truncation=True, max_length=256).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_length=256)
print(f"{sample}")

print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
