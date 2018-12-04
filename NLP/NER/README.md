# Named Entity Recognition with Tensorflow

This repo implements a NER model using Tensorflow (LSTM + CRF + chars embeddings).


## Task

Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
上海        综合 指数 日K线图 ( 图片 )
B-LOCATION O    O   O      O O   O
```


## Model

Similar to [Lample et al.](https://arxiv.org/abs/1603.01360) and [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (sgns here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF



## Getting started


1. Download the Chinese word vectors from

```
https://pan.baidu.com/s/14JP1gD7hcmsWdSpTvA3vKA
```

2. Build the training data, train and evaluate the model with
```
bash run.sh
```


## Details


Here is the breakdown of the commands executed in `run.sh`:

1. [Skipped] Build vocab from the data and extract trimmed sgns vectors according to the config in `model/config.py`.

```
python build_data.py
```

2. Train the model with

```
python train.py
```


3. Evaluate and interact with the model with
```
python evaluate.py
```


Data iterators and utils are in `model/data_utils.py` and the model with training/test procedures is in `model/ner_model.py`

Training time on NVidia GTX 1080Ti is 610 seconds per epoch on train set using character embeddings and CRF.
