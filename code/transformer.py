import math
import numpy as np
import tensorflow as tf

class AttentionMatrix(tf.keras.layers.Layer):

    def __init__(self, *args, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]  # window size of queries
        window_size_keys    = K.get_shape()[1]  # window size of keys

        mask_vals = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        mask = tf.convert_to_tensor(value=mask_vals, dtype=tf.float32)
        atten_mask = tf.tile(tf.reshape(mask, [-1, window_size_queries, window_size_keys]), [tf.shape(input=K)[0], 1, 1])

        d_k = tf.cast(K.get_shape()[2], tf.float32)
        scores = (tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k))
        if self.use_mask == True:
            scores = scores + atten_mask
        scores = tf.nn.softmax(scores)
        return scores


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        self.K = self.add_weight("K", shape=[input_size, output_size])
        self.Q = self.add_weight("Q", shape=[input_size, output_size])
        self.V = self.add_weight("V", shape=[input_size, output_size])
        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        K = tf.tensordot(inputs_for_keys, self.K, 1)
        V = tf.tensordot(inputs_for_values, self.V, 1)
        Q = tf.tensordot(inputs_for_queries, self.Q, 1)

        scores = self.attn_mtx((K, Q))
        weighted = tf.matmul(scores, V)
        return weighted


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        self.h1 = AttentionHead(emb_sz, int(emb_sz / 3), use_mask)
        self.h2 = AttentionHead(emb_sz, int(emb_sz / 3), use_mask)
        self.h3 = AttentionHead(emb_sz, int(emb_sz / 3), use_mask)
        self.linear = tf.keras.layers.Dense(emb_sz)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        o1 = self.h1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        o2 = self.h2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        o3 = self.h3(inputs_for_keys, inputs_for_values, inputs_for_queries)
        concat = tf.concat([o1, o2, o3], axis=-1)
        out = self.linear(concat)
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        #self.ff_layer = tf.keras.layers.Dense(emb_sz)
        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation="relu"),
            tf.keras.layers.Dense(emb_sz)
        ])

        self.self_atten         = AttentionHead(emb_sz, emb_sz, True)  if not multiheaded else MultiHeadedAttention(emb_sz, True)
        self.self_context_atten = AttentionHead(emb_sz, emb_sz, False) if not multiheaded else MultiHeadedAttention(emb_sz, False)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def call(self, inputs, context_sequence):
        #masked_attn = self.self_atten(inputs, inputs, inputs)
        #masked_attn += inputs
        #masked_attn = self.layer_norm(masked_attn)
        #unmasked_attn = self.self_context_atten(context_sequence, context_sequence, inputs)
        #unmasked_attn += masked_attn
        #unmasked_attn = self.layer_norm(unmasked_attn)
        #ff = self.ff_layer(unmasked_attn)
        #ff += unmasked_attn
        #ff = self.layer_norm(ff)
        #out = tf.nn.relu(ff)
        #return out
        atten_out = self.self_atten(inputs, inputs, inputs)
        atten_norm = self.layer_norm(atten_out + inputs)
        context_atten_out = self.self_context_atten(context_sequence, context_sequence, atten_norm)
        atten_norm = self.layer_norm(context_atten_out + atten_norm)

        ff_out = self.ff_layer(atten_norm)
        ff_out += atten_norm
        ff_norm = self.layer_norm(ff_out)
        return tf.nn.relu(ff_norm)


def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]      # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    angle_rates = 1 / (10000 ** depths)               # (1, depth)
    angle_rads = positions * angle_rates              # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        #self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        #self.pos_encoding = positional_encoding(window_size, embed_size)
        self.pos_encoding = positional_encoding(length=window_size, depth=embed_size)[..., :window_size, :]

    def call(self, x):
        #length = tf.shape(x)[1]
        #x = self.embedding(x)
        #x *= tf.math.sqrt(tf.cast(self.embed_size, dtype=tf.float32))
        #x = x + self.pos_encoding[:length, :]
        x = self.embedding(x)
        factor = tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        #return x
        return x * factor + self.pos_encoding
    