This repository contains files of a Capstone 2 project of [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by [Alexey Grigorev](https://github.com/alexeygrigorev)

# Overview

# Overview

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

To reproduce the results:

1. **Data Access**: The dataset is available in [Kaggle's Sentiment and Emotion Analysis Dataset](https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset). You can directly download or use the notebook available in the repository to access to it.
2. **Execution**: Run the Colab notebook (`notebook.ipynb`) or the training script (`train.py`) without errors.

## Environment Setup

To set up the project, follow these steps.

1. Clone the repository at your desired path, then open the folder:
   ```bash
      git clone https://github.com/yasamangs/Text-Emotion-Classification.git
      cd Text-Emotion-Classification
   ```
2. Create a Conda environment and activate it:
   ```bash
      conda create --name Text-Emotion-env python=3.8
      conda activate Text-Emotion-env
   ```
3. Install the required dependencies:
   ```bash
      pip install -r requirements.txt
   ```
Or, use Google Colab and upload the notebook directly on it.

## Running the Model Training Script

To train the model using the provided script, execute:
   ```bash
      python train.py
   ```
## Running the streamlit app

## The Streamlit App

To interact with the trained emotion classification model and visualize predictions, a **Streamlit app** is provided. The app allows users to input text and get real-time predictions for the emotion conveyed by the sentence. The app is built using **Streamlit**, a popular Python library for creating web applications for machine learning models.

### Steps to Run the Streamlit App Locally

Follow the steps below to run the app on your local machine:

1. Install Dependencies:

Before running the app, make sure all the required dependencies are installed. These include **Streamlit**, **Transformers**, and other packages used for data processing and model inference.

To install all dependencies, run the following command:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit App:
   
Once the dependencies are installed, navigate to the directory containing the app.py file and run the following command:
```bash
streamlit run app.py
```

3. Interacting with the App:

After running the above command, the app will open in your default web browser at [Localhost](http://localhost:8501). The user interface will allow you to:

- Enter text: Type a sentence in the provided input field.
- Analyze: The model will predict the emotion conveyed by the text.
- View Prediction: The predicted emotion is displayed along with the logits (the raw model output) and the final predicted emotion.

## Model Deployment

The fine-tuned BERT model is deployed on **Hugging Face** for easy access and real-time predictions. The deployment steps include:
   - Uploading the fine-tuned model to the **Hugging Face Model Hub**.
   - Integrating the model with **Streamlit** for a user-friendly web interface.
   - Deploying the model on **Streamlit Cloud**, allowing users to interact with the model and make emotion predictions.

## Real-time Predictions

Once deployed, users can input a sentence, and the model will predict the associated emotion. This deployment approach ensures scalability and easy access for real-time applications.


# License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
