import struct
import numpy as np
import torch
import torch.nn as nn

RANDOMIZED = False # Mettre à True si vous n'avez pas accès au fichier Word2Vec

print("Computing vocabulary...")
vocabulary = set()
with open("/mnt/ssd/flickr30k/results_20130124.token", encoding="utf-8") as f:
    for line in f:
        _, words = line.strip().split("\t")
        for word in words.split(" "):
            vocabulary.add(word)
print("Number of words:", len(vocabulary))

def read_until(open_file):
    while True:
        char = open_file.read(1)
        if char == b' ':
            break
        yield char

def read_word2vec():
    with open("/mnt/ssd/word2vec/GoogleNews-vectors-negative300.bin", "rb") as f:
        first_line = f.readline()
        n_words, n_dim = map(int, first_line.split())
        for n_word in range(n_words):
            word = b''.join(read_until(f)).decode('utf-8', errors='ignore')
            vector = struct.unpack('f' * n_dim, f.read(n_dim * 4))
            yield word, vector

def create_embedding_layer(vocab, randomized=False):
    if randomized:
        idx_to_word = [""]
        word_to_idx = {"": 0}
        weights = []
        for i, word in enumerate(vocab):
            idx_to_word.append(word)
            word_to_idx[word] = i + 1
            weights.append(np.random.rand(1, 300))
    else:
        weights = []
        idx_to_word = []
        word_to_idx = dict()
        idx = 0
        for word, vector in read_word2vec():
            if word in vocab or idx == 0:
                idx_to_word.append(word)
                word_to_idx[word] = idx
                weights.append(np.array([vector]))
                idx += 1
    n_words = len(weights)
    n_dim = len(weights[0][0])
    weights = np.concatenate(weights, axis=0)
    emb_layer = nn.Embedding(n_words, n_dim)
    emb_layer.load_state_dict({"weight": torch.Tensor(weights)})
    if not randomized:
        emb_layer.requires_grad = False
    return emb_layer, idx_to_word, word_to_idx

def tokenize(sentence, word2idx, max_length=20):
    res = [0] # 0 est souvent l'indice de début/fin ou padding
    sentence = sentence.lower()
    for word in sentence.split():
        if word in word2idx:
            res.append(word2idx.get(word, 0))
    if len(res) > max_length:
        return res[:max_length]
    return res + [0] * (max_length - len(res))

def untokenize(tokens, idx2word):
    return [idx2word[x] for x in tokens]

pretrained_embeddings, idx2word, word2idx = create_embedding_layer(vocabulary, RANDOMIZED)
print("Vocabulary size:", len(idx2word))
