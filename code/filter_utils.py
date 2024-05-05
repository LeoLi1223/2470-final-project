from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


def print_captions_and_label(captions, task):
  MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
  tokenizer = AutoTokenizer.from_pretrained(MODEL)

  # download label mapping
  labels=[]
  mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
  with urllib.request.urlopen(mapping_link) as f:
      html = f.read().decode('utf-8').split("\n")
      csvreader = csv.reader(html, delimiter='\t')
  labels = [row[1] for row in csvreader if len(row) > 1]

  tf_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
  tf_model.save_pretrained(MODEL)

  


  for caption in captions:
    print(caption)
    encoded_input = tokenizer(caption, return_tensors='tf')
    output = tf_model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)

    # ranking
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
    print()