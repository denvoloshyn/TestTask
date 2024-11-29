"""### 1. Load a dataset

<span style="font-size:16px">For the indentification of mountain names inside the text we use [Few-NERD](http://ningding97.github.io/fewnerd) dataset, which is also available at [Kaggle](http://www.kaggle.com/datasets/nbroad/fewnerd). More exactly we use a supervised part of this dataset.</span>
"""

from datasets import load_dataset

fewnerd = load_dataset('json', data_files={
    'train': '/input/fewnerd/supervised/train.json',
    'val': '/input/fewnerd/supervised/dev.json',
    'test': '/input/fewnerd/supervised/test.json',
})
fewnerd

"""<span style="font-size:16px">Loading tag dictionaries</span>"""

import json

with open("/input/fewnerd/id2coarse_tags.json", "r") as f:
    id2coarse_tag = json.load(f)
print(id2coarse_tag)

with open("/input/fewnerd/id2fine_tags.json", "r") as f:
    id2fine_tag = json.load(f)
id2fine_tag

MOUTAIN_TAG = 24

rows_with_mountain_tag = [i for i, row in enumerate(fewnerd["train"]["fine_tags"]) if MOUTAIN_TAG in row]
len(rows_with_mountain_tag), rows_with_mountain_tag[:5]

"""<span style="font-size:16px">Some examples from the train dataset:</span>"""

for x in fewnerd["train"].select(rows_with_mountain_tag[:5]):
    print(x, "\n")

"""### 2. Preprocess

<span style="font-size:16px">Replacing tags other than mountain tag with 0, mountain tags with 1, and removing extra columns from new datasets</span>
"""

def tag_map(tag):
    return 1 if tag == MOUTAIN_TAG else 0

def tag_list_map(tag_list):
    return list(map(tag_map, tag_list))

def fine_tags_map(examples):
    examples["mountain_tags"] = list(map(tag_list_map, examples["fine_tags"]))
    return examples

fewnerd_mountains = fewnerd.map(fine_tags_map, remove_columns=["coarse_tags", "fine_tags", "id"], batched=True)

"""<span style="font-size:16px">Some examples from the processed train dataset:</span>"""

for x in fewnerd_mountains["train"].select(rows_with_mountain_tag[:5]):
    print(x, "\n")

"""<span style="font-size:16px">Computing number of mountain and O tags in processed datasets and their proportion</span>"""

def print_mountain_dataset_stat(name, dataset):
    mountain_tags_num = 0
    tags_num = 0
    for tags in dataset["mountain_tags"]:
        mountain_tags_num += sum(tags)
        tags_num += len(tags)
    o_tags_num = tags_num - mountain_tags_num
    print(f"{name:<5} dataset - mountain tags: {mountain_tags_num}, O tags: {o_tags_num}, proportion: {mountain_tags_num/o_tags_num}")

for k in fewnerd_mountains.keys():
    print_mountain_dataset_stat(k, fewnerd_mountains[k])

"""<span style="font-size:16px">Load DistilBERT tokenizer to preprocess the tokens field</span>"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

"""<span style="font-size:16px">An example of tokenization in action:</span>"""

example = fewnerd_mountains["train"]["tokens"][75]
tokenized_input = tokenizer(example, is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

print(example, "\n")
print(tokenized_input, "\n")
print(tokens)

"""<span style="font-size:16px">Tokenizer adds some special tokens [CLS] and [SEP] and the subword tokenization creates a mismatch between the input and labels. A single word corresponding to a single label may now be split into two subwords. We realign the tokens and labels and remove extra columns from new datasets.</span>"""

# The value that is ignored and does not contribute to the input gradient in CrossEntropyLoss
IGNORE_INDEX = -100

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["mountain_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i) # Map tokens to their respective word
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(IGNORE_INDEX) # Set the special tokens to IGNORE_INDEX
            else:
                label_ids.append(label[word_idx]) # Label each token of a given word
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

fewnerd_mountains = fewnerd_mountains.map(tokenize_and_align_labels, remove_columns=["tokens", "mountain_tags"], batched=True)

"""<span style="font-size:16px">An example from the processed train dataset:</span>"""

print(fewnerd_mountains["train"][75])

"""<span style="font-size:16px">Set a data collator that will dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length</span>"""

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

"""### 3. Train

<span style="font-size:16px">Before we start training a model, create a list of lables and dictionaries of label ids and labels</span>
"""

label_list = [id2fine_tag[str(0)], id2fine_tag[str(MOUTAIN_TAG)]]
print(label_list, "\n")

id2label = {i: label for i, label in enumerate(label_list)}
print(id2label, "\n")

label2id = {label: i for i, label in enumerate(label_list)}
print(label2id)

"""<span style="font-size:16px">Load DistilBERT model with AutoModelForTokenClassification along with the number of expected labels, and the label mappings</span>"""

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

"""<span style="font-size:16px">Create a function that computes metrics from predictions and labels, ignoring labels for special tokens</span>"""

import numpy as np
import evaluate

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != IGNORE_INDEX]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [ [label_list[l] for l in label if l != IGNORE_INDEX] for label in labels ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

"""<span style="font-size:16px">Due to the imbalance of mountain tags number and O tags number in the datasets, we want to use class weights in the loss function. For this we need a customization of Trainer class.</span>"""

import torch
from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, tag_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag_weights = tag_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Compute custom loss
        weight=torch.tensor(self.tag_weights)
        if torch.cuda.is_available():
           weight = weight.cuda()
        loss_fun = torch.nn.CrossEntropyLoss(weight)
        loss = loss_fun(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

"""<span style="font-size:16px">Set parameters for training the model, create an instance of CustomTrainer, train the model, and evaluate it on the test dataset</span>"""

from transformers import TrainingArguments

tag_weights = [0.1, 1]

training_args = TrainingArguments(
    output_dir="train_output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    report_to="none",
)

trainer = CustomTrainer(
    model=model,
    tag_weights=tag_weights,
    args=training_args,
    train_dataset=fewnerd_mountains["train"],
    eval_dataset=fewnerd_mountains["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(fewnerd_mountains["test"])

"""<span style="font-size:16px">Save the best model and tokenizer to the specified directory</span>"""

save_dir = "fewnerd-mountains-model-train"

trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)