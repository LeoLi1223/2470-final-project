from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

class Groberta:
    def __init__(self):
        task='offensive'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # download label mapping
        # self.labels=[]
        # mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        # with urllib.request.urlopen(mapping_link) as f:
        #     html = f.read().decode('utf-8').split("\n")
        #     csvreader = csv.reader(html, delimiter='\t')
        # self.labels = [row[1] for row in csvreader if len(row) > 1]

        # TF
        self.model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
        self.model.save_pretrained(MODEL)
    
    def compute_offensive_score(self, captions: list[str]):
        out = []
        for caption in captions:
            encoded_input = self.tokenizer(caption, return_tensors='tf')
            output = self.model(encoded_input)
            scores = output[0][0].numpy()
            scores = softmax(scores)
            out.append(scores[1])

        return out # offensive scores
