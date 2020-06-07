# %%
import re 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import os
import tokenizers 
import transformers
from tqdm import tqdm
from transformers import RobertaConfig, TFRobertaModel
# %%
from keras.layers import Input, Dropout, Flatten, Activation
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
# %%
MAX_LENGTH = 128
ROBERTA_PATH='roberta-base'
 # %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.dropna()
train.reset_index(inplace = True)
train['sentiment'] = train['sentiment'].astype('category')
tqdm.pandas()

# %%
def clean_text(text, remove_stopwords = False):

    if(remove_stopwords):
        stop = set(stopwords.words('english'))
        text = [w for w in word_tokenize(text) if (w not in stop)]
        text = ' '.join(w for w in text)

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?:\S+|www.\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    return text
# %%
train['ctext'] = train['text'].apply(lambda s: clean_text(s))
train['cselected_text'] = train['selected_text'].apply(lambda s: clean_text(s)) 

# %%
n_rows = train.shape[0]
input_ids = np.ones((n_rows, MAX_LENGTH), dtype='int32')
attention_mask  = np.zeros((n_rows, MAX_LENGTH), dtype='int32')


# %%
tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_file='vocab-roberta-base.json',
                                             merges_file='merges-roberta-base.txt',
                        
                                             lowercase=True,
                                             add_prefix_space=True)


# %%
sentiment_ids = {'negative': 2430, 'positive': 1313, 'neutral': 7974}

for k in range(train.shape[0]):
    
    tokens_na_string = []
    idx = 0
    text1 = train.loc[k, 'ctext']
    text2 = train.loc[k, 'cselected_text']
    idx = text1.find(text2)
    resposta = np.zeros((len(text1)))
    resposta[idx:idx+len(text2)] = 1
    enc = tokenizer.encode(text1)

    for id in enc.ids:
        w = tokenizer.decode([id])
        tokens_na_string.append((idx, idx+len(w)))
        idx += len(w)

    start_end = []
    for i, (a,b) in enumerate(tokens_na_string):
        sm = np.sum(resposta[a:b])
        if sm>1: start_end.append(i)

    s_id = sentiment_ids[train.loc[k, 'sentiment']]
    input_ids[k, :len(enc.ids) + 5] = [0] + enc.ids + [2,2] + [s_id] + [2]
    attention_mask[k, :len(enc.ids) + 5] = 1


# %%

def iniciar_modelo():
    ids = Input((MAX_LEN,), dtype=tf.int32)
    att = Input((MAX_LEN,), dtype=tf.int32)
    
    config = RobertaConfig.from_pretrained('config-roberta-base.json')
    roberta_model = TFRobertaModel.from_pretrained('pretrained-roberta-base.h5', config = config)

    x = roberta_model(ids, attention_mask=att)

    x1 = Dropout(0.1)(x[0])
    x1 = Conv1D(1,1)(x1)
    x1 = Flatten()(x1)
    x1 = Activation('softmax')(x1)

    x2 = Dropout(0.1)(x[0])
    x2 = Conv1D(1,1)(x2)
    x2 = Flatten()(x2)
    x2 = Activation('softmax')(x2)

    model = Model(inputs = [ids, att], outputs = [x1, x2])
    otimizador = Adam(learning_rate = 2e-5)
    model.compile(loss='categorical_crossentropy', optimizer=otimizador)

    return model
# %%
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if ((len(a) == 0) and (len(b) == 0)): return 0.5
    c = a.intersection(b)
    return float((len(c))/ (len(a) + len(b) - len(c))) 