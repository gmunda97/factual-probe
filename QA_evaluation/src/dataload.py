from transformers import BertTokenizerFast, default_data_collator
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def preprocess_data(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    answers = examples["answers"]
    
    # Tokenize question and context together
    tokenized_examples = tokenizer(questions, contexts, truncation=True, padding="max_length", max_length=384, return_offsets_mapping=True)
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        # Loop through each example
        answer = answers[i]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        
        # Find start and end token index for the current span
        token_start_index = 0
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        
        token_end_index = token_start_index
        while token_end_index < len(offsets) and offsets[token_end_index][1] <= end_char:
            token_end_index += 1
        
        # If the answer cannot be found in the text, then CLS index is used
        start_positions.append(token_start_index - 1 if token_start_index > 0 else 0)
        end_positions.append(token_end_index - 1 if token_end_index > 0 else 0)
    
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

squad_dataset = load_dataset("squad")
tokenized_datasets = squad_dataset.map(preprocess_data, batched=True)

train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=default_data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=default_data_collator)

# count number of data points in train and validation datasets
print(len(train_dataset))
print(len(val_dataset))

# save the dataset to disk
train_dataset.save_to_disk('./../resources/data/train')
val_dataset.save_to_disk('./../resources/data/val')