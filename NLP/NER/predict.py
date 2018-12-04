from model.ner_model import NERModel
from model.config import Config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    with open(config.filename_test, 'r') as rf, open(config.filename_predict, 'w') as wf:
        for line in rf.readlines():
            words = line.strip().split()
            tags = model.predict(words)
            modified_tags = ['B-TIME' if (word.endswith('年') or word.endswith('月') or word.endswith('日'))
                                         and word[:-1].isdigit() else tag for word, tag in zip(words, tags)]
            wf.write(' '.join(modified_tags)+'\n')
#    with open(config.filename_train, 'r') as rf, open('data/train.prediction.txt', 'w') as wf:
#        for line in rf.readlines():
#            words = [ls.split('/')[0] for ls in line.strip().split()]
#            wf.write(' '.join(model.predict(words))+'\n')
    with open(config.filename_dev, 'r') as rf, open('results/dev.prediction.txt', 'w') as wf:
        for line in rf.readlines():
            words = [ls.split('/')[0] for ls in line.strip().split()]
            tags = model.predict(words)
            modified_tags = ['B-TIME' if (word.endswith('年') or word.endswith('月') or word.endswith('日'))
                                         and word[:-1].isdigit() else tag for word, tag in zip(words, tags)]
            wf.write(' '.join(modified_tags) + '\n')


if __name__ == "__main__":
    main()
