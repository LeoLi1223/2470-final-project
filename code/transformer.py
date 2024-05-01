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

        atten_matrix = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(K.get_shape()[2], dtype=tf.float32))
        if self.use_mask == True:
            atten_matrix += atten_mask
        atten_matrix = tf.nn.softmax(atten_matrix)
        return atten_matrix


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        # TODO:
        # Initialize the weight matrices for K, V, and Q.
        # They should be able to multiply an input_size vector to produce an output_size vector
        # Hint: use self.add_weight(...)
        self.K = self.add_weight(shape=(input_size, output_size), name="K")
        self.Q = self.add_weight(shape=(input_size, output_size), name="Q")
        self.V = self.add_weight(shape=(input_size, output_size), name="V")
        self.attn_mtx = AttentionMatrix(use_mask=self.use_mask)


    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        STUDENT MUST WRITE:

        This functions runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """

        # TODO:
        # - Apply 3 matrix products to turn inputs into keys, values, and queries. 
        # - You will need to use tf.tensordot for this.
        # - Call your AttentionMatrix layer with the keys and queries.
        # - Apply the attention matrix to the values.

        K = tf.tensordot(inputs_for_keys, self.K, axes=1)
        V = tf.tensordot(inputs_for_values, self.V, axes=1)
        Q = tf.tensordot(inputs_for_queries, self.Q, axes=1)

        atten_mtx = self.attn_mtx((K, Q))
        out = tf.matmul(atten_mtx, V)

        return out


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, **kwargs):
        super(MultiHeadedAttention, self).__init__(**kwargs)

        ## TODO: Add 3 heads as appropriate and any other necessary components
        self.h1 = AttentionHead(emb_sz, emb_sz//3, True)
        self.h2 = AttentionHead(emb_sz, emb_sz//3, True)
        self.h3 = AttentionHead(emb_sz, emb_sz//3, True)
        self.ff = tf.keras.layers.Dense(emb_sz)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        TODO: FOR CS2470 STUDENTS:

        This functions runs a multiheaded attention layer.

        Requirements:
            - Splits data for 3 different heads of size embed_sz/3
            - Create three different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_values: tensor of [batch_size x KEY_WINDOW_SIZE x input_size ]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size ]
        :return: tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size ]
        """
        # print("inputs_for_keys", inputs_for_keys.shape)
        # print("inputs_for_values", inputs_for_values.shape)
        # print("inputs_for_queries", inputs_for_queries.shape)
        o1 = self.h1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        o2 = self.h2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        o3 = self.h3(inputs_for_keys, inputs_for_values, inputs_for_queries)
        # print(o1.shape)
        # print(o2.shape)
        # print(o3.shape)
        concat = tf.concat([o1, o2, o3], -1)
        # print(concat.shape)
        out = self.ff(concat)
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, multiheaded=False, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.ff_layer = tf.keras.layers.Dense(emb_sz)

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
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
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
    