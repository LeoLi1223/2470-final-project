import os
import argparse
import numpy as np
import pickle
import tensorflow as tf
from typing import Optional
from types import SimpleNamespace
from read_glove import get_glove_embedding


from model import ImageCaptionModel, accuracy_function, loss_function
from decoder import TransformerDecoder, RNNDecoder
import transformer


def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type',           required=True,              choices=['rnn', 'transformer'],     help='Type of model to train')
    parser.add_argument('--task',           required=True,              choices=['train', 'test', 'both'],  help='Task to run')
    parser.add_argument('--data',           required=True,              help='File path to the assignment data file.')
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=300,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=20,     help='Window size of text entries.')
    parser.add_argument('--chkpt_path',     default='',                 help='where the model checkpoint is')
    parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

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

def main(args):

    ##############################################################################
    ## Data Loading
    with open(args.data, 'rb') as data_file:
        data_dict = pickle.load(data_file)

    feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 50, axis=0)
    # img_prep  = lambda x: np.repeat(x, 5, axis=0)
    train_captions  = np.array(data_dict['train_captions'])
    test_captions   = np.array(data_dict['test_captions'])
    train_img_feats = feat_prep(data_dict['train_image_features'])
    test_img_feats  = feat_prep(data_dict['test_image_features'])
    # train_images    = img_prep(data_dict['train_images'])
    # test_images     = img_prep(data_dict['test_images'])
    word2idx        = data_dict['word2idx']
    # idx2word        = data_dict['idx2word']

    ##############################################################################
    ## Training Task
    if args.task in ('train', 'both'):
        ##############################################################################
        ## Model Construction
        decoder_class = {
            'rnn'           : RNNDecoder,
            'transformer'   : TransformerDecoder
        }[args.type]

        decoder = decoder_class(
            vocab_size  = len(word2idx), 
            hidden_size = args.hidden_size, 
            window_size = args.window_size,
            embedding_matrix = get_embedding_matrix(word2idx)
        )
        
        model = ImageCaptionModel(decoder)
        compile_model(model, args)
        train_model(
            model, train_captions, train_img_feats, word2idx['<pad>'], args, 
            valid = (test_captions, test_img_feats)
        )
        if args.chkpt_path: 
            ## Save model to run testing task afterwards
            save_model(model, args)
                
    ##############################################################################
    ## Testing Task
    if args.task in ('test', 'both'):
        if args.task != 'both': 
            ## Load model for testing. Note that architecture needs to be consistent
            model = load_model(args)
        if not (args.task == 'both' and args.check_valid):
            test_model(model, test_captions, test_img_feats, word2idx['<pad>'], args)

    ##############################################################################

##############################################################################
## UTILITY METHODS

def save_model(model, args):
    '''Loads model based on arguments'''
    os.makedirs(f"{args.chkpt_path}", exist_ok=True)
    # print(model)
    # print(args.chkpt_path)
    tf.keras.models.save_model(model, args.chkpt_path)
    print(f"Model saved to {args.chkpt_path}")


def load_model(args):
    '''Loads model by reference based on arguments. Also returns said model'''
    model = tf.keras.models.load_model(
        args.chkpt_path,
        custom_objects=dict(
            AttentionHead           = transformer.AttentionHead,
            AttentionMatrix         = transformer.AttentionMatrix,
            MultiHeadedAttention    = transformer.MultiHeadedAttention,
            TransformerBlock        = transformer.TransformerBlock,
            PositionalEncoding      = transformer.PositionalEncoding,
            TransformerDecoder      = TransformerDecoder,
            RNNDecoder              = RNNDecoder,
            ImageCaptionModel       = ImageCaptionModel
        ),
    )
    
    ## Saving is very nuanced. Might need to set the custom components correctly.
    ## Functools.partial is a function wrapper that auto-fills a selection of arguments. 
    ## so in other words, the first argument of ImageCaptionModel.test is model (for self)
    from functools import partial
    model.test    = partial(ImageCaptionModel.test,    model)
    model.train   = partial(ImageCaptionModel.train,   model)
    model.compile = partial(ImageCaptionModel.compile, model)
    model.get_filtered_captions = partial(ImageCaptionModel.get_filtered_captions, model)
    model.get_unfiltered_captions = partial(ImageCaptionModel.get_unfiltered_captions, model)
    model.get_offensive_score = partial(ImageCaptionModel.get_offensive_score, model)
    compile_model(model, args)
    print(f"Model loaded from '{args.chkpt_path}'")
    return model


def compile_model(model, args):
    '''Compiles model by reference based on arguments'''
    optimizer = tf.keras.optimizers.get(args.optimizer).__class__(learning_rate = args.lr)
    model.compile(
        optimizer   = optimizer,
        loss        = loss_function,
        metrics     = [accuracy_function]
    )


def train_model(model, captions, img_feats, pad_idx, args, valid):
    '''Trains model and returns model statistics'''
    stats = []
    try:
        for epoch in range(args.epochs):
            stats += [model.train(captions, img_feats, pad_idx, batch_size=args.batch_size)]
            if args.check_valid:
                prp, acc = model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
                print(f"\r[epoch {epoch}]\t avg_prp:{prp}\t avg_acc:{acc}", end="\n")
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else: 
            raise e
        
    return stats


def test_model(model, captions, img_feats, pad_idx, args):
    '''Tests model and returns model statistics'''
    perplexity, accuracy = model.test(captions, img_feats, pad_idx, batch_size=args.batch_size)
    return perplexity, accuracy


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())
