from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Define the path to the checkpoint you want to load
checkpoint_path = './checkpoint-16830'

# Load the model from the checkpoint
model = BertForSequenceClassification.from_pretrained(checkpoint_path)

# load the tokenizer (for text processing)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# write a sample
sample_sentence = "write whatever sentence you want"

# Tokenize the sample sentence
inputs = tokenizer(sample_sentence, return_tensors="pt",
                   padding=True, truncation=True, max_length=64)

# Perform inference
outputs = model(**inputs)
logits = outputs.logits

d = {
    0: 'anger',
    1: 'fear',
    2: 'joy',
    3: 'love',
    4: 'sad',
    5: 'suprise'
}

y_pred = np.argmax(outputs.logits.detach().numpy(), axis=1)
