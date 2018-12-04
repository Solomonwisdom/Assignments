import pickle
import os
import random
import numpy as np
from pathlib import Path


def load_tags(filename):
    d = dict()
    with open(filename) as f:
        for idx, tag in enumerate(f):
            d[tag.strip()] = idx
    return d


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        for word_and_label in line.strip().split():
            word, label = word_and_label.strip().split('/')
            sent_.append(word)
            tag_.append(label)
        data.append((sent_, tag_))
        sent_, tag_ = [], []
    return data


def is_eng(word):
    for ch in word:
        if not (('\u0041' <= ch <= '\u005a') or ('\u0061' <= ch <= '\u007a')):
            return False
    return True


def vocab_build(vocab_path, tag_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    tags = set()
    for sent_, tag_ in data:
        for tag in tag_:
            tags.add(tag)
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif is_eng(word):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    if min_count > 0:
        low_freq_words = []
        for word, [_, word_freq] in word2id.items():
            if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
                low_freq_words.append(word)
        for word in low_freq_words:
            del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    with open(tag_path, 'w') as fw:
        fw.write('\n'.join(tags))

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []

    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif is_eng(word):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def pre_embedding(word_to_idx, embedding_path):
    embeddings = np.zeros((len(word_to_idx), 300))
    # Get relevant glove vectors
    found = 0
    print('Reading sgns file (may take a while)')
    with Path('data_path/merge_sgns_bigram_char300.txt').open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, len(word_to_idx)))
    np.savez_compressed(embedding_path, embeddings=embeddings)


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

