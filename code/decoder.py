import tensorflow as tf

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        # define feed forward layer(s)
        #self.image_embedding = tf.keras.layers.Dense(self.hidden_size)
        self.image_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(hidden_size)
        ])

        # define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)

        # define decoder layer    
        self.decoder = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)

        # define classification layer(s) 
        #self.classifier = tf.keras.layers.Dense(self.vocab_size)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(vocab_size)
        ])

    def call(self, encoded_images, captions):
        encoded_images = self.image_embedding(encoded_images)
        captions = self.embedding(captions)
        decoded = self.decoder(captions, initial_state=(encoded_images, tf.zeros_like(encoded_images)))
        probs = self.classifier(decoded)
        return probs

########################################################################################

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # define feed forward layer(s) 
        #self.image_embedding = tf.keras.layers.Dense(hidden_size)
        self.image_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(hidden_size)
        ])

        # define positional encoding
        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # define transformer decoder layer
        self.decoder = TransformerBlock(emb_sz=hidden_size, multiheaded=True)

        # define classification layer(s)
        #self.classifier = tf.keras.layers.Dense(vocab_size)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(vocab_size)
        ])

    def call(self, encoded_images, captions):
        encoded_images = self.image_embedding(tf.expand_dims(encoded_images, 1))
        captions = self.encoding(captions)
        decoded = self.decoder(captions, encoded_images)
        probs = self.classifier(decoded)
        return probs
    
"""     def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'window_size': self.window_size,
        })
        return config """
