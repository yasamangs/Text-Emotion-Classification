# Text-Emotion-Classification

This repository contains files of a Capstone 2 project of [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by [Alexey Grigorev](https://github.com/alexeygrigorev)

# Overview

## Overview

The project aims to classify text into various emotion categories such as **anger**, **fear**, **joy**, **love**, **sadness**, and **surprise**. The project leverages a **fine-tuned BERT model** to perform emotion classification from text data. The model is trained on the **Sentiment and Emotion Analysis Dataset** from Kaggle, which contains sentences labeled with corresponding emotions.

### Project Features
- **BERT model** fine-tuned for **emotion classification**.
- **Streamlit** interface for easy interaction with the model.
- **Hugging Face Hub** deployment for accessibility and sharing.
- Deployment on **Streamlit Cloud**, ensuring scalability and public access.

### Use Cases
This project can be extended for a variety of applications:
- **Sentiment analysis** in social media and customer feedback.
- **Emotion recognition** in customer support and chatbots.
- **Psychological analysis** for understanding emotional tone in text.

By using **Hugging Face** and **Streamlit Cloud**, this project demonstrates an end-to-end solution for deploying machine learning models with minimal effort and high scalability.

## Dataset

The dataset used in this project is derived from the [Kaggle's Sentiment and Emotion Analysis Dataset](https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset). The dataset contains text samples labeled with various emotions. The dataset includes the following features:

| Feature     | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| **sentence** | The text or sentence that is being classified for emotion.                  |
| **emotion**  | The label for the emotion conveyed by the text.                              |

The possible labels are: 
- `anger`                                                                    
- `fear`                                                                     
- `joy`                                                                      
- `love`                                                                     
- `sad`                                                                      
- `surprise`   

### Dataset Files
The dataset consists of the **combined_emotion.csv** file, which contains the sentences and their corresponding emotion labels.

### Example Data
Here’s a small sample from the `combined_emotion.csv` file:

| sentence                        | emotion |
|----------------------------------|---------|
| I feel really happy today        | joy     |
| I am feeling quite sad           | sad     |
| I’m so angry with how things turned out | anger |
| I am afraid of what might happen next | fear |

### Preprocessing
The dataset is preprocessed by:
- Removing duplicate entries.
- Handling missing values.
- Tokenizing and lemmatizing the sentences using the **spaCy** library.

The processed data is saved as a **Pickle file** (`emotion_data.pkl`) in the project, which contains the cleaned and tokenized sentences ready for model training.

## Exploratory Data Analysis (EDA)

An extensive **EDA** was conducted to understand the dataset and identify key insights:

- **Data Overview**: Analyzed the basic structure of the dataset, checked for missing values, and assessed data types. We explored the two primary features: `sentence` (the text input) and `emotion` (the target label).
  
- **Target Variable Analysis**: We examined the distribution of emotions in the dataset to understand the balance between classes and identify any potential issues related to class imbalance.
  
- **Feature Relationships**: Investigated the relationship between sentence length and emotion classes to identify if longer sentences tend to belong to certain emotions more than others.
  
- **Feature Importance**: We explored the distribution of sentence lengths and visualized how various emotion classes were distributed across different sentence lengths, and how these features might impact the model's prediction.

