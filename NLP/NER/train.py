from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # create instance of config
    config = Config()
    # build model
    model = NERModel(config)
    model.build()
    if os.path.exists("results/test/model.weights/"):
        model.restore_session("results/test/model.weights/")  # optional, restore weights
        model.reinitialize_weights("proj")

    # create datasets
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main()
