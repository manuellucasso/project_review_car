# Car Reviews NLP Project

This project uses Hugging Face Transformers and NLP pipelines to analyze car reviews. It performs multiple natural language processing tasks, including:

- Sentiment analysis
- Translation
- Question answering
- Text summarization

## Features

### 1. Sentiment Analysis
Classifies car reviews as **positive** or **negative** using a pre-trained sentiment analysis model.

### 2. Translation
Translates a car review from English to Spanish using a translation model.

### 3. Question Answering
Answers a question based on the content of a car review.

### 4. Summarization
Generates a short summary of a car review using a summarization model.

## Technologies Used

- Python
- Pandas
- PyTorch
- Hugging Face Transformers
- Evaluate

## Dataset

The project reads data from:

- `data/car_reviews.csv`

It also uses a reference translation file:

- `data/reference_translations.txt`

## How It Works

### Sentiment Analysis
The script loads a pre-trained model:

```python
distilbert-base-uncased-finetuned-sst-2-english
