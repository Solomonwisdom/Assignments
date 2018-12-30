# -*- coding:utf-8 -*-
import argparse
import numpy as np
import os
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=5, help='the number of topics')
parser.add_argument('--n_top', type=int, default=10, help='the number of words to show')
parser.add_argument('--max_iter', type=int, default=100, help='the number of words to show')
opt = parser.parse_args()
n_components = opt.n_components
n_top_words = opt.n_top
max_iter = opt.max_iter

data_samples = []
idx_to_word = list()
word_to_idx = dict()
current_idx = 0
if os.path.isfile("processed_news.txt"):
    with open("processed_news.txt", "r") as f:
        for line in f.readlines():
            words = line.strip().split(" ")
            ids = []
            for word in words:
                if word in word_to_idx:
                    ids.append(word_to_idx[word])
                else:
                    ids.append(current_idx)
                    word_to_idx[word] = current_idx
                    idx_to_word.append(word)
                    current_idx += 1
            data_samples.append(ids)
else:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    stopworddic = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    # Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
    # to filter out useless terms early on: the posts are stripped of headers,
    # footers and quoted replies, and common English words, words occurring in
    # only one document or in at least 95% of the documents are removed.

    print("Loading dataset...")
    t0 = time.time()
    # set flag to allow verbose regexps
    # abbreviations, e.g. U.S.A.
    # numbers, incl. currency and percentages
    # words w/ optional internal hyphens/apostrophe
    # ellipsis
    # special characters with meanings
    pattern = r"""(?x)
    (?:[A-Z]\.)+
    |\d+(?:\.\d+)?%?
    |\w+(?:[-']\w+)*
    |\.\.\.
    |(?:[.,;"'?():-_`])
    """


    def is_english(word):
        flag = True
        for uchar in word:
            if uchar > u'\u007f':
                flag = False
        return flag

    def check_word(word, stopworddic):
        if len(word) == 1 and not word.isalpha():
            return False
        return is_english(word.lower()) and word.lower() not in stopworddic


    with open("news.txt", "r") as f, open("processed_news.txt", "w") as pf:
        for i, line in enumerate(f):
            tokens = nltk.regexp_tokenize(line.strip(), pattern)
            words = [wordnet_lemmatizer.lemmatize(token.lower())
                     for token in tokens if check_word(token, stopworddic)]
            if len(words) > 0:
                ids = []
                for word in words:
                    if word in word_to_idx:
                        ids.append(word_to_idx[word])
                    else:
                        ids.append(current_idx)
                        word_to_idx[word] = current_idx
                        idx_to_word.append(word)
                        current_idx += 1
                data_samples.append(ids)
                pf.write(" ".join(words)+'\n')
            # if i % 1000 == 0:
            #     print(i)
        print("done in {0:.3f}s.".format(time() - t0))


def initialize(docs):
    # Initialization with Online Gibbs Sampling
    for d, doc in enumerate(docs):
        tw = []
        for word in doc:
            p_t = np.divide(np.multiply(ntd[:, d], nwt[word, :]), nt)
            t = np.random.multinomial(1, p_t / p_t.sum()).argmax()
            tw.append(t)
            ntd[t][d] = ntd[t][d] + 1
            nwt[word, t] = nwt[word, t] + 1
            nt[t] = nt[t] + 1
        twd.append(np.array(tw))


def gibbs_iteration(docs):
    # Collapsed Gibbs Sampling Iteration
    for d, doc in enumerate(docs):
        for w, word in enumerate(doc):
            # Decrement counts for old topic of the word
            t = twd[d][w]
            ntd[t][d] = ntd[t][d] - 1
            nwt[word, t] = nwt[word, t] - 1
            nt[t] = nt[t] - 1

            # Sample new topic
            p_t = np.divide(np.multiply(ntd[:, d], nwt[word, :]), nt)
            t = np.random.multinomial(1, p_t / p_t.sum()).argmax()

            # Increment counts for new topic of the word
            twd[d][w] = t
            ntd[t][d] = ntd[t][d] + 1
            nwt[word, t] = nwt[word, t] + 1
            nt[t] = nt[t] + 1


def perplexity(docs):
    nd = np.sum(ntd, 0)
    n = 0
    ll = 0.0
    for d, doc in enumerate(docs):
        for word in enumerate(doc):
            ll = ll + np.log(((nwt[word, :] / nt) * (ntd[:, d] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


def print_top_words(topics, feature_names, n_top_words):
    with open("mylda/K{}.txt".format(n_components), 'w') as f:
        for topic_idx, topic in enumerate(topics):
            message = "Topic #{}: ".format(topic_idx)
            message += " ".join(["{0:s}: {1:.2f}%".format(feature_names[i], topic[i] * 100 / topic.sum())
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
            f.write(message+'\n')
    print()


alpha = 5
beta = 0.1

# topic of word w in doc d
twd = []
# number of words of topic t in doc d
ntd = np.zeros((n_components, len(data_samples))) + alpha
# number of times word w is in topic t
nwt = np.zeros((len(word_to_idx), n_components)) + beta
# number of words in topic t
nt = np.zeros(n_components) + (len(word_to_idx) * beta)

print("Initialize Gibbs Sampling...")
t0 = time()
initialize(data_samples)
print("done in {0:.3f}s.".format(time() - t0))
print()

for i in range(max_iter):
    print("Iteration {}...".format(i + 1))
    t0 = time()
    gibbs_iteration(data_samples)
    print("done in {0:.3f}s.".format(time() - t0))

print()
print_top_words(nwt.T, idx_to_word, n_top_words)
