import pandas as pd
from datasets import Dataset

train_df = pd.read_parquet("/content/drive/MyDrive/test.parquet")
test_df = pd.read_parquet("/content/drive/MyDrive/test.parquet")
train_data = Dataset.from_pandas(train_df)
test_data = Dataset.from_pandas(test_df)

def preprocess_function(sample, padding="max_length"):
    model_inputs = tokenizer(sample["human"], max_length=256, padding=padding, truncation=True)
    labels = tokenizer(sample["assistant"], max_length=256, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized_dataset = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
test_tokenized_dataset = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)
