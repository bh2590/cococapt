import nltk
import dill as pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import numpy as np
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
import logging
logger = logging.getLogger("Vocab builder")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class SpecialTokens:
    def __init__(self):
        self.END = '<end>'
        self.OOV = "<oov>"
        self.START = "<start>"
        self.PAD = '<pad>'


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.spec_tokens= SpecialTokens()

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def loadWordVectors(self, filepath="/home/hanozbhathena/project/data/glove.840B.300d.txt", 
                        dimensions=300):
        """Read pretrained GloVe vectors"""
        dic = {}
        words= []
        with open(filepath) as ifs:
            for line in ifs:
                line = line.strip()
                if not line:
                    continue
                row = line.split()
                token = row[0]
                if token not in self.word2idx:
                    continue
                try:
                    data = [float(x) for x in row[1:]]
                except ValueError:
                    words.append(' '.join(row[:-300]))
                    continue
                if len(data) != dimensions:
                    raise RuntimeError("wrong number of dimensions")
                dic[token] = np.asarray(data)
        logging.info("{} lines were not meeting split standards: {}".format(len(words), words))
        dic[self.spec_tokens.START]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
        dic[self.spec_tokens.END]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
        dic[self.spec_tokens.OOV]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
        dic[self.spec_tokens.PAD]= np.random.uniform(low=-0.25, high=0.25, size=dimensions)
        
        self.word_embeddings = np.zeros((len(self.word2idx), dimensions))
        oov_count= 0
        for word, ind in self.word2idx.items():
            try:
                self.word_embeddings[ind]= dic[word]
            except KeyError:
                oov_count+=1
                self.word_embeddings[ind]= dic[self.spec_tokens.OOV]
        logging.info("{} words from COCO were OOV for Glove".format(oov_count))
        logging.info("word vectors loaded using" + filepath)


def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word(vocab.spec_tokens.PAD)
    vocab.add_word(vocab.spec_tokens.END)
    vocab.add_word(vocab.spec_tokens.OOV)
    vocab.add_word(vocab.spec_tokens.START)

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    
    #Assign the Glove word embeddings
    vocab.loadWordVectors()
    
    assert vocab.word2idx[vocab.spec_tokens.PAD] == 0, "Make PAD index equal to 0 for convenience"
    
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='/home/hanozbhathena/project/data/annotations/captions_train2017.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='/home/hanozbhathena/project/data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)