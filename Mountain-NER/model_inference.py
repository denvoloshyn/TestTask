"""
<span style="font-size:16px">Load the model and tokenizer from the specified path and define a function that tags each word in a text with either the mountain tag or the O tag</span>
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_path = "fewnerd-mountains-model"

model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Returns a list of word and tag pairs based on the model and tokenizer
def get_word_tag_list(text):
    tokenized_input = tokenizer(text, return_tensors="pt", truncation=True)

    # Compute a list of predicted tags for all tokens based on the model
    with torch.no_grad():
        logits = model(**tokenized_input).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_tags = [model.config.id2label[t.item()] for t in predictions[0]]

    # List mapping token IDs to word IDs
    word_ids = tokenized_input.word_ids()

    # Get a list mapping word IDs to token IDs
    word_to_token_ids = []
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id >= len(word_to_token_ids):
                word_to_token_ids.append([])
            word_to_token_ids[word_id].append(idx)

    # Generate a list of word and tag pairs
    word_tag_list = []
    for word_id in range(len(word_to_token_ids)):
        span = tokenized_input.word_to_chars(word_id)
        word = text[span.start:span.end]

        token_id = word_to_token_ids[word_id][0]
        tag = predicted_tags[token_id]

        word_tag_list.append((word, tag))

    return word_tag_list

"""<span style="font-size:16px">Examples of the model output:</span>"""

# Prints the model output
def print_word_tag_list(text):
    word_tag_list = get_word_tag_list(text)
    for p in word_tag_list:
        print(f"{p[0]} : {p[1]}")

text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
print_word_tag_list(text)

text = """
The Mont Blanc massif is popular for outdoor activities like hiking, climbing, trail running and winter sports like skiing, and snowboarding.
The most popular climbing route to the summit of Mont Blanc is the Goûter Route, which typically takes two days.
"""
print_word_tag_list(text)

text = "Mont Blanc is a beautiful rooftop cafe."
print_word_tag_list(text)