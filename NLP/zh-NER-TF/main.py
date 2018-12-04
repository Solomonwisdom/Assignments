import tensorflow as tf
import numpy as np
import os
import argparse
import time
import model as md
from utils import str2bool, get_logger
from data import read_corpus, read_dictionary, load_tags, random_embedding, vocab_build, pre_embedding


# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--build', type=str, default='no')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--predict_data', type=str, default='test', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=2, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='pretrain', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/predict')
parser.add_argument('--demo_model', type=str, default='1543732999', help='model for test and demo')
args = parser.parse_args()


# get char embeddings
if args.build == 'yes':
    vocab_build(os.path.join('.', args.train_data, 'word2id.pkl'),
                os.path.join('.', args.train_data, 'tags.txt'), os.path.join('.', args.train_data, 'train.txt'), 0)
    word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
    pre_embedding(word2id, os.path.join('.', args.train_data, 'merge_sgns_bigram_char300d.trimmed.npz'))
else:
    word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
tag2label = load_tags(os.path.join('.', args.train_data, 'tags.txt'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = os.path.join('.', args.train_data, 'merge_sgns_bigram_char300d.trimmed.npz')
    embeddings = np.load(embedding_path)["embeddings"]


# read corpus and get training data
if args.mode != 'predict':
    train_path = os.path.join('.', args.train_data, 'train.txt')
    test_path = os.path.join('.', args.test_data, 'dev.txt')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)


# paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path):
    os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


# training model
if args.mode == 'train':
    model = md.BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    # train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

# testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = md.BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

# demo
elif args.mode == 'predict':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = md.BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess, \
            open(os.path.join('.', args.train_data, args.predict_data+'.content.txt')) as pf:
        print('============= predict =============')
        saver.restore(sess, ckpt_file)
        model.predict(sess, pf.readlines(), os.path.join('.', args.train_data, args.predict_data+'.prediction.txt'))
