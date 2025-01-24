import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import spacy
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize, remove stopwords, and lemmatize using spaCy
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(words)


emotion_data = pd.read_csv("./combined_emotion.csv")

nlp = spacy.load("en_core_web_sm")

emotion_data_cleaned = emotion_data.drop_duplicates()

emotion_sentences = emotion_data_cleaned.iloc[:, 0].apply(preprocess_text)
emotion_labels = emotion_data_cleaned.iloc[:, 1]


# Encode the labels
label_encoder = LabelEncoder()
emotion_labels_encoded = label_encoder.fit_transform(emotion_labels)

# Check the class distribution of encoded labels
data = pd.DataFrame({'sentence': emotion_sentences,
                    'label': emotion_labels_encoded})

# Find the largest class size
class_distribution = data['label'].value_counts()

# Find the smallest class size (to target it for balancing)
minority_class_size = class_distribution.min()

# Perform under-sampling to balance the classes: Resample majority class
# Define the target size for the majority classes
target_majority_size = minority_class_size

# Separate the classes
balanced_data = []

# For each class, resample the majority class (or use the minority class size)
for label, group in data.groupby('label'):
    if len(group) > minority_class_size:
        # If the class is larger than the smallest class, downsample it
        downsampled = resample(group, replace=False,
                               n_samples=minority_class_size, random_state=42)
        balanced_data.append(downsampled)
    else:
        # If the class is already smaller or equal to the minority class, keep it as is
        balanced_data.append(group)

# Combine the downsampled majority class with the minority class
balanced_data = pd.concat(balanced_data)

# Shuffle the resulting dataset to avoid any bias
balanced_data = balanced_data.sample(
    frac=1, random_state=42).reset_index(drop=True)

# Get the balanced sentences and labels
balanced_sentences = balanced_data['sentence'].tolist()
balanced_labels = balanced_data['label'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# let's split the data into train, validation, and test sets
emotion_X_train, emotion_X_test, emotion_y_train, emotion_y_test = train_test_split(
    balanced_sentences, balanced_labels, test_size=0.2, random_state=42
)

# Further split the training data into train and validation sets
emotion_X_train, emotion_X_val, emotion_y_train, emotion_y_val = train_test_split(
    emotion_X_train, emotion_y_train, test_size=0.2, random_state=42
)

max_len = 64  # Limit sequence length to 64 tokens
train_dataset = EmotionDataset(
    balanced_sentences, balanced_labels, tokenizer, max_len=max_len)
val_dataset = EmotionDataset(
    emotion_X_val, emotion_y_val, tokenizer, max_len=max_len)

# Step 3: Train model with the reduced dataset and sequence length
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=len(label_encoder.classes_))

training_args = TrainingArguments(
    output_dir='./results/',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',  # Change to accuracy for best model selection
    report_to="none",  # Disable W&B logging
)

# Updated compute_metrics function for accuracy
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: {'accuracy': accuracy_score(
        p.label_ids, p.predictions.argmax(axis=-1))}  # Use accuracy here
)

# Train the model
trainer.train()

# Save the trained model to Google Drive
model.save_pretrained('./checkpoint-16830')
