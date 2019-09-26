# -*- coding: utf-8 -*-
# @Time    : 2019-08-19 14:10
# @Author  : finupgroup
# @FileName: ShowAndTell.py
# @Software: PyCharm

from keras.models import Model
import pandas as pd
from keras.preprocessing import sequence
from keras.layers import *
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import keras
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import pickle
import os


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x


def get_data(token_file_path='flickr30k-token/results_20130124.token', ins_ratio=0.8):
    annotations = pd.read_table(token_file_path, sep='\t', header=None,
                                names=['image', 'caption'])

    data_ = []
    pbar = tqdm(total=annotations.shape[0])
    for x, y in tqdm(zip(annotations['image'], annotations['caption'])):
        data_.append((inception_model.predict(preprocess('flickr30k-images/' + x.split('#')[0])).reshape(-1, ),
                      '<start> ' + y + ' <end>'))
        pbar.update(1)

    train_data_ = data_[:int((len(data_) / 5) * ins_ratio) * 5]
    valid_data_ = data_[int((len(data_) / 5) * ins_ratio) * 5:]

    caps = []
    for x in train_data_:
        caps.append(x[1])

    words = [i.split() for i in caps]
    unique_ = []
    for i in words:
        unique_.extend(i)

    unique_ = list(set(unique_))
    word2idx_ = {val: index + 1 for index, val in enumerate(unique_)}
    idx2word_ = {index + 1: val for index, val in enumerate(unique_)}

    max_len_ = 0
    for c in caps:
        c = c.split()
        if len(c) > max_len_:
            max_len_ = len(c)
    return data_, train_data_, valid_data_, unique_, word2idx_, idx2word_, max_len_


def get_data_split(ele):
    data_split = []
    for i in range(len(ele)):
        tmp = ele[i][1]
        for j in range(len(tmp.split()) - 1):
            tamp = []
            for txt in tmp.split()[:j + 1]:
                try:
                    tamp.append(word2idx[txt])
                except KeyError:
                    tamp.append(0)
            # Initializing with zeros to create a one-hot encoding matrix
            # This is what we have to predict
            # Hence initializing it with vocab_size length
            next_words = np.zeros(len(unique) + 1)
            # Setting the next word to 1 in the one-hot encoded matrix
            try:
                next_words[word2idx[tmp.split()[j + 1]]] = 1
            except KeyError:
                next_words[0] = 1

            data_split.append((ele[i][0], tamp, next_words))
    return data_split


class data_generator:
    def __init__(self, data_, batch_size=128):
        self.data = data_
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            ids = list(range(len(self.data)))
            np.random.shuffle(ids)
            _X1, _X2, _Y = [], [], []
            for i in ids:
                d = self.data[i]
                img = d[0]
                x = d[1][:max_len]
                y = d[2]

                _X1.append(img)
                _X2.append(x)
                _Y.append(y)
                _X1_array = np.array(_X1)
                _X2_array = np.array(_X2)
                _Y_array = np.array(_Y)

                if len(_X1) == self.batch_size or i == ids[-1]:
                    _X2_array = sequence.pad_sequences(_X2_array, maxlen=max_len, padding='post')
                    yield [[_X1_array, _X2_array], _Y_array]
                    _X1, _X2, _Y = [], [], []


def predict_captions(img_file_path):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = inception_model.predict(preprocess(img_file_path)).reshape(-1, )
        pred = model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(pred[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > max_len:
            break

    return ' '.join(start_word[1:-1])


def beam_search_predictions(img_file_path, beam_index=3):
    start = [word2idx["<start>"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = inception_model.predict(preprocess(img_file_path)).reshape(-1, )
            pred = model.predict([np.array([e]), np.array(par_caps)])

            word_pred = np.argsort(pred[0])[-beam_index:]

            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_pred:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += pred[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


def get_bleu_score(reference, candidate):
    """
    Calculate BLEU score (Bilingual Evaluation Understudy) from
    Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.
    "BLEU: a method for automatic evaluation of machine translation."
    In Proceedings of ACL. http://www.aclweb.org/anthology/P02-1040.pdf
    """
    score = sentence_bleu(reference, candidate)
    return score


inception = InceptionV3(weights='imagenet')
new_input = inception.input
hidden_layer = inception.layers[-2].output
inception_model = Model(new_input, hidden_layer)

if 'data.p' in os.listdir('.'):
    # with open("data.p", "wb") as encoded_pickle:
    #     pickle.dump((data, train_data, valid_data, unique, word2idx, idx2word, max_len), encoded_pickle)
    data, train_data, valid_data, unique, word2idx, idx2word, max_len = pickle.load(open('data.p', 'rb'))
else:
    data, train_data, valid_data, unique, word2idx, idx2word, max_len = get_data()

train_split = get_data_split(train_data)
valid_split = get_data_split(valid_data)

# 输入为图片抽取的特征
inputs1 = Input(shape=(2048,))
# 经过一层全连接层
x1 = Dense(300, activation='relu')(inputs1)
# 复制max_len次
x1 = RepeatVector(max_len)(x1)

# 输入部分描述
inputs2 = Input(shape=(max_len,))
# 进行词嵌入
x2 = Embedding(len(unique) + 1, 300, input_length=max_len, mask_zero=True)(inputs2)
# 经过LSTM层，并保留每个时间步的输出
x2 = LSTM(256, return_sequences=True)(x2)
# 对每个时间步的输出，分别应用一层全连接层
x2 = TimeDistributed(Dense(200, activation='relu'))(x2)

# 我们将两部分处理后的输入在最后一维进行拼接
x3 = Concatenate()([x1, x2])
# 再经过LSTM，输出最后一个时间步
x3 = Bidirectional(LSTM(256, return_sequences=False))(x3)
# 将全连接层的结果进行softmax，映射到词汇表每个词的概率
outputs = Dense(len(unique) + 1, activation='softmax')(x3)

model = Model([inputs1, inputs2], outputs)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()


train_D = data_generator(train_split)
valid_D = data_generator(valid_split)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    callbacks=[keras.callbacks.ModelCheckpoint("weights-{epoch:04d}--{val_acc:.4f}.h5", monitor='val_acc',
                                               save_best_only=True, verbose=1)]
    # callbacks=[TensorBoard(log_dir='./logs', write_graph=True, write_images=True)]
)
