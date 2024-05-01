import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
#from read_glove import get_glove_embedding

def preprocess_captions(captions, window_size):
    for i, caption in enumerate(captions):
        caption = caption.replace('<sep> ', '')
        caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
        clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
        caption_new = ['<start>'] + clean_words[:window_size-1] + ['<end>']
        captions[i] = caption_new

def get_image_features(image_names, data_folder, vis_subset=100):
    image_features = []
    vis_images = []
    inception = tf.keras.applications.InceptionV3(False)  
    gap = tf.keras.layers.GlobalAveragePooling2D()  
    pbar = tqdm(image_names)
    for i, image_name in enumerate(pbar):
        img_path = f'{data_folder}/images/{image_name}.jpg'
        pbar.set_description(f"[({i+1}/{len(image_names)})] Processing '{img_path}' into 2048-D Inception V3 GAP Vector")
        with Image.open(img_path) as img:
            img_array = np.array(img.resize((300, 300)))
        img_in = tf.keras.applications.inception_v3.preprocess_input(img_array)[np.newaxis, :]
        image_features += [gap(inception(img_in))]
        if i < vis_subset:
            vis_images += [img_array]
    print()
    return image_features, vis_images

def load_data(data_folder):
    text_file_path = f'{data_folder}/captions.txt' 

    with open(text_file_path) as file:
        examples = file.read().splitlines() 

    # map each image name to a list containing all 3000 of its captions
    image_names_to_captions = {}
    simplify = lambda text: re.sub(r"[-_]+", "-", re.sub(r"[^\w\s-]+", "", text.lower().strip().replace("_", "-")).replace(" ", "-").replace("ñ", ""))
    for example in examples:
        l = example.split("\t")
        img_name, caption = simplify(l[0]), l[2]
        image_names_to_captions[img_name] = image_names_to_captions.get(img_name, []) + [caption]

    # randomly split examples into training and testing sets
    shuffled_images = list(image_names_to_captions.keys())
    random.seed(0)
    random.shuffle(shuffled_images)
    train_image_names = shuffled_images[60:]
    test_image_names = shuffled_images[:60]

    def get_all_captions(image_names):
        to_return = []
        for image in image_names:
            captions = image_names_to_captions[image]
            for caption in captions:
                to_return.append(caption)
        return to_return

    # get lists of all the captions in the training and testing sets
    train_captions = get_all_captions(train_image_names)
    test_captions = get_all_captions(test_image_names)
    
    # remove special characters and other necessary preprocessing
    window_size = 20 
    preprocess_captions(train_captions, window_size)
    preprocess_captions(test_captions, window_size)

    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for caption in train_captions:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'

    unk_captions(train_captions, 3) 
    unk_captions(test_captions, 3)
    
    # count the number of <unk>
    def count_unk(caption):
        return caption.count("<unk>")

    def get_part_captions(captions, n=50):
        top_captions = []
        for start in range(0, len(captions), 3000):
            end = start + 3000
            caps = captions[start:end]
            sorted_caps = sorted(caps, key=lambda cap: count_unk(cap))
            top_captions.extend(sorted_caps[:n])
        return top_captions

    train_captions = get_part_captions(train_captions)
    test_captions = get_part_captions(test_captions)

    # pad captions so they all have equal length
    def pad_captions(captions, window_size):
        for caption in captions:
            caption += (window_size + 1 - len(caption)) * ['<pad>'] 

    pad_captions(train_captions, window_size)
    pad_captions(test_captions, window_size)

    # assign unique ids to every word left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for caption in train_captions:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    if "<unk>" not in word2idx.keys():
        word2idx["<unk>"] = vocab_size
        vocab_size += 1
    for caption in test_captions:
        for index, word in enumerate(caption):
            if word not in word2idx:
                caption[index] = word2idx["<unk>"]
            else:
                caption[index] = word2idx[word]
                
    # extract image features
    print("Getting training embeddings")
    train_image_features, train_images = get_image_features(train_image_names, data_folder)
    print("Getting testing embeddings")
    test_image_features, test_images = get_image_features(test_image_names, data_folder)

    return dict(
        train_captions          = np.array(train_captions),
        test_captions           = np.array(test_captions),
        train_image_features    = np.array(train_image_features),
        test_image_features     = np.array(test_image_features),
        train_images            = np.array(train_images),
        test_images             = np.array(test_images),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )
    

def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')

if __name__ == '__main__':
    data_folder = '../memes900k'
    create_pickle(data_folder)