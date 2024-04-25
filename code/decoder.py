import tensorflow as tf
from read_glove import get_glove_embedding
import numpy as np

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, embedding_matrix, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.embedding = embedding_matrix

        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers

        # Define feed forward layer(s) to embed image features into a vector 
        # with the models hidden size
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size)

        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size, trainable=True) ##TODO: replace with GloVE
        self.embedding.build((1,))
        self.embedding.set_weights([embedding_matrix])

        # Define decoder layer that handles language and image context:     
        self.decoder = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(self.vocab_size)

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder 
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        img_embd = self.image_embedding(encoded_images)
        sentence_embd = self.embedding(captions)
        decoding = self.decoder(sentence_embd, initial_state=(img_embd, tf.zeros_like(img_embd)))
        logits = self.classifier(decoding)
        return logits


########################################################################################

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, embedding_matrix, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer(s) to embed image features into a vector 
        self.image_embedding = tf.keras.layers.Dense(hidden_size)

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(hidden_size, multiheaded=True)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(vocab_size)

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate **logits**
        encoded_images = self.image_embedding(tf.expand_dims(encoded_images, 1))
        captions = self.encoding(captions)
        decoded = self.decoder(captions, encoded_images)
        logits = self.classifier(decoded)
        return logits
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'window_size': self.window_size,
        })
        return config
