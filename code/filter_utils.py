from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import os

from groberta import Groberta


def print_captions_and_label(captions: list[str], task: str):
  MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
  tokenizer = AutoTokenizer.from_pretrained(MODEL)

  # download label mapping
  labels=[]
  mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
  with urllib.request.urlopen(mapping_link) as f:
      html = f.read().decode('utf-8').split("\n")
      csvreader = csv.reader(html, delimiter='\t')
  labels = [row[1] for row in csvreader if len(row) > 1]

  # TF
  model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
  model.save_pretrained(MODEL)

  for caption in captions:
    print(caption)
    text = caption
    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)

    print(labels[0], scores[0])
    print(labels[1], scores[1])

    # ranking
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
    print()

def second(text):
    model = Groberta()
    score = model.compute_offensive_score(text)
    print(score)

if __name__ == "__main__":
    # print_captions_and_label(["hey girl, you are mean", "hey girl, you are good", "hey girl, you are terrible"], "offensive")
    second(["hey girl, you are mean", "hey girl, you are good", "hey girl, you are terrible"])