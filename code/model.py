import numpy as np
import tensorflow as tf
import pickle
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from groberta import Groberta


task='offensive'
MODEL_OFFENSIVE = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL_OFFENSIVE)

class ImageCaptionModel(tf.keras.Model):

    def __init__(self, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    @tf.function
    def call(self, encoded_images, captions):
        return self.decoder(encoded_images, captions)  

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self, train_captions, train_image_features, padding_index, batch_size=30):
        total_loss = total_seen = total_correct = 0
        assert len(train_captions) == len(train_image_features)
        shuffled_indices = tf.random.shuffle(tf.range(0, len(train_captions)))
        train_captions = tf.gather(train_captions, shuffled_indices)
        train_image_features = tf.gather(train_image_features, shuffled_indices)
        for index, end in enumerate(range(batch_size, len(train_captions)+1, batch_size)):
            start = end - batch_size
            batch_image_features = train_image_features[start:end, :]
            decoder_input = train_captions[start:end, :-1]
            decoder_labels = train_captions[start:end, 1:]

            with tf.GradientTape() as tape:
                probs = self(batch_image_features, decoder_input)
                mask = decoder_labels != padding_index
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = self.loss_function(probs, decoder_labels, mask)
                accuracy = self.accuracy_function(tf.cast(probs, tf.int64), tf.cast(decoder_labels, tf.int64), mask)
            
            ## update weights
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

        avg_loss = float(total_loss / total_seen)
        avg_acc = float(total_correct / total_seen)
        avg_prp = np.exp(avg_loss)
        return avg_loss, avg_acc, avg_prp

    def test(self, test_captions, test_image_features, padding_index, batch_size=30):
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = test_image_features[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')      
        return avg_prp, avg_acc

    def get_offensive_score(self, text):
        # download label mapping
        task = 'offensive'
        labels=[]
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]

        # TF
        model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_OFFENSIVE)
        model.save_pretrained(MODEL_OFFENSIVE)

        encoded_input = tokenizer(text, return_tensors='tf')
        output = model(encoded_input)
        scores = output[0][0].numpy()
        scores = softmax(scores)

        return scores[1]

    def get_filtered_captions(self, image_embedding, wordToIds, padID, window_length):
        """
        Function used to generate a caption using an ImageCaptionModel given
        an image embedding. 
        """
        temp = 0.05
        while temp < 0.2:
            idsToWords = {id: word for word, id in wordToIds.items()}
            unk_token = wordToIds['<unk>']
            caption_so_far = [wordToIds['<start>']]
            while len(caption_so_far) < window_length and caption_so_far[-1] != wordToIds['<end>']:
                caption_input = np.array([caption_so_far + ((window_length - len(caption_so_far)) * [padID])])
                logits = self(np.expand_dims(image_embedding, 0), caption_input)
                logits = logits[0][len(caption_so_far) - 1]
                probs = tf.nn.softmax(logits / temp).numpy()
                next_token = unk_token
                attempts = 0
                while next_token == unk_token and attempts < 5:
                    next_token = np.random.choice(len(probs), p=probs)
                    attempts += 1
                caption_so_far.append(next_token)
            out = []
            for x in caption_so_far:
              word = idsToWords[x]
              if word == "sep":
                word = "<newline>"
              elif word == "emp":
                continue
              out.append(word)
            text = ' '.join(out[1:-1])
            # text = ' '.join([idsToWords[x] for x in caption_so_far][1:-1])
            offensive_score = self.get_offensive_score(text)
            if offensive_score > 0.5:
                temp = temp + 0.05
            else:
                return text, offensive_score
        return text, offensive_score
    
    def get_unfiltered_captions(self, image_embedding, wordToIds, padID, window_length):
        """
        Function used to generate a caption using an ImageCaptionModel given
        an image embedding. 
        """
        temp = 0.05
        idsToWords = {id: word for word, id in wordToIds.items()}
        unk_token = wordToIds['<unk>']
        caption_so_far = [wordToIds['<start>']]
        while len(caption_so_far) < window_length and caption_so_far[-1] != wordToIds['<end>']:
            caption_input = np.array([caption_so_far + ((window_length - len(caption_so_far)) * [padID])])
            logits = self(np.expand_dims(image_embedding, 0), caption_input)
            logits = logits[0][len(caption_so_far) - 1]
            probs = tf.nn.softmax(logits / temp).numpy()
            next_token = unk_token
            attempts = 0
            while next_token == unk_token and attempts < 5:
                next_token = np.random.choice(len(probs), p=probs)
                attempts += 1
            caption_so_far.append(next_token)
        out = []
        for x in caption_so_far:
          word = idsToWords[x]
          if word == "sep":
            word = "<newline>"
          elif word == "emp":
            continue
          out.append(word)
        text = ' '.join(out[1:-1])
        # text = ' '.join([idsToWords[x] for x in caption_so_far][1:-1])
        offensive_score = self.get_offensive_score(text)
        return text, offensive_score   

    def get_config(self):
        base_config = super().get_config()
        config = {
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop("decoder")
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        return cls(decoder, **config)


def accuracy_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Computes the batch accuracy

    :param prbs:  float tensor, word prediction probabilities [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
    :param labels:  integer tensor, word prediction labels [BATCH_SIZE x WINDOW_SIZE]
    :param mask:  tensor that acts as a padding mask [BATCH_SIZE x WINDOW_SIZE]
    :return: scalar tensor of accuracy of the batch between 0 and 1
    """
    correct_classes = tf.argmax(prbs, axis=-1) == labels
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def loss_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Calculates the model cross-entropy loss after one forward pass
    Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

    :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :param mask:  tensor that acts as a padding mask [batch_size x window_size]
    :return: the loss of the model as a tensor
    """
    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss