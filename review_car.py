# Import necessary packages
import pandas as pd
import torch

from transformers import logging
from transformers import pipeline
import evaluate
logging.set_verbosity(logging.WARNING)

# Loading data
df = pd.read_csv("data/car_reviews.csv", sep=";")

#1 Classify car reviews

#Define the sentiment analysis pipeline using a pre-trained model
classifier = pipeline(task="sentiment-analysis",model='distilbert-base-uncased-finetuned-sst-2-english')

predicted_labels = []

for review in df["Review"]:
    result=classifier(review)
    predicted_labels.append(result[0])

print(predicted_labels) 

#mapping sentiment prediction
predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]

#mapping sentiment reference 
sentiment_reference = [1 if label == "POSITIVE" else 0 for label in df['Class']]

print(sentiment_reference)
accuracy=evaluate.load("accuracy")
f1_score=evaluate.load("f1")

accuracy_result=accuracy.compute(predictions=predictions,references=sentiment_reference)
f1_result=f1_score.compute(predictions=predictions,references=sentiment_reference)
print(accuracy_result)
print(f1_result)


#2 Translate a car review

#Define the translation pipeline using a pre-trained model
translator=pipeline(task="translation", model = "Helsinki-NLP/opus-mt-en-es")

#First Review
first_review = df["Review"].iloc[0]

with open("data/reference_translations.txt", "r") as file:
    reference_translation = file.read().splitlines()

print(reference_translation)    

#Translating only the first two sentences
translated_review = translator(first_review, max_length=28)[0]['translation_text']

evaluator_bleu = evaluate.load("bleu")

result_bleu = evaluator_bleu.compute(predictions=[translated_review],references=[reference_translation])

bleu_score = result_bleu['bleu']

#3 Q&A

#Define the question-answering pipeline using a pre-trained model
q_and_a=pipeline(task="question-answering", model = "deepset/minilm-uncased-squad2")

#define the question
question = "What did he like about the brand?"
context = df["Review"].iloc[1]

result = q_and_a(question=question, context=context)

answer = result["answer"]

print(answer)

#4 Summarize

#Define the summarization pipeline using a pre-trained model
summarizer = pipeline(task="summarization",model="facebook/bart-large-cnn")

last_review=df["Review"].iloc[-1]

summary=summarizer(last_review,max_length=55,min_length=20)

print(summary[0]['summary_text'])
