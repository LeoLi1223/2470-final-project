import tensorflow as tf
from read_glove import get_glove_embedding
import numpy as np

def get_embedding_matrix(word2idx):
    embeddings_index = get_glove_embedding("../GloVE/glove.6B.300d.txt")
    num_tokens = len(word2idx)
    embedding_dim = 300
    hits = 0
    misses = 0
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "<start>" and "<end>" and "<unk>" and "<pad>"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            embedding_matrix[i] = tf.random.normal((embedding_dim, ))
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix

def gen_captions(model, image_embeddings, wordToIds, padID, temp, window_length):
    """
    Wrapper for gen_caption_temperature in order to generate multiple captions.
    """
    ret = []
    for image_embedding in image_embeddings:
        caption = gen_caption_temperature(model, image_embedding, wordToIds, padID, temp, window_length)
        ret.append(caption)
    return ret

def gen_caption_temperature(model, image_embedding, wordToIds, padID, temp, window_length):
    """
    Function used to generate a caption using an ImageCaptionModel given
    an image embedding.
    """
    idsToWords = {id: word for word, id in wordToIds.items()}
    unk_token = wordToIds['<unk>']
    caption_so_far = [wordToIds['<start>']]
    while len(caption_so_far) < window_length and caption_so_far[-1] != wordToIds['<end>']:
        caption_input = np.array([caption_so_far + ((window_length - len(caption_so_far)) * [padID])])
        logits = model(np.expand_dims(image_embedding, 0), caption_input)
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
    return ' '.join(out[1:-1])