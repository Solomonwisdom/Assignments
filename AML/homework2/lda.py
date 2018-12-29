from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import os

n_components = 10
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join(["{0:s}: {1:.2f}%".format(feature_names[i], topic[i]*100/topic.sum())
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


data_samples = []
if os.path.isfile("processed_news.txt"):
    with open("processed_news.txt", "r") as f:
        for line in f.readlines():
            data_samples.append(line)
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
    t0 = time()
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


    with open("news.txt", "r") as f, open("processed_news.txt", "w") as pf:
        for i, line in enumerate(f):
            tokens = nltk.regexp_tokenize(line.strip(), pattern)
            words = [wordnet_lemmatizer.lemmatize(token.lower())
                     for token in tokens if is_english(token.lower()) and token.lower() not in stopworddic]
            if len(words) > 0:
                data_samples.append(" ".join(words))
                pf.write(" ".join(words)+'\n')
            # if i % 1000 == 0:
            #     print(i)
    print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

n_features = len(tf_vectorizer.get_feature_names())
n_samples = len(data_samples)
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=25,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
