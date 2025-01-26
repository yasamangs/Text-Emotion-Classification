# Text-Emotion-Classification

This repository contains files of a Capstone 2 project of [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by [Alexey Grigorev](https://github.com/alexeygrigorev)

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
