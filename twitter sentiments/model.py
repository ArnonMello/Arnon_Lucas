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
from transformers import *

# %%
MAX_LENGTH = 96
ROBERTA_PATH='roberta-base'
# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.dropna()
train['sentiment'] = train['sentiment'].astype('category')


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
tokes = np.zeros((n_rows, MAX_LENGTH), dtype='int32')
start_token = np.zeros((n_rows, MAX_LENGTH), dtype='int32')
end_token = np.zeros((n_rows, MAX_LENGTH), dtype='int32')



# %%
tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_file='vocab-roberta-base.json',
                                             merges_file='merges-roberta-base.txt',
                        
                                             lowercase=True,
                                             add_prefix_space=True)

# %%

enc = tokenizer.encode("I like to draw circles!")

# %%
t = enc.ids[0]
w = tokenizer.decode([t])
print(w)

# %%
text1 = "arnon is nice, he plays games with me because i'm bad"
text2 = "arnon is nice"
idx = text1.find(text2)
resposta = np.zeros((len(text1)))
resposta[idx:idx+len(text2)] = 1

enc = tokenizer.encode(text1)

# %%

for i, id in enumerate(enc.ids):
    print(tokenizer.decode([id]), id, i)
    print('\n')


# %%
tokens_na_string = []
idx = 0


for id in enc.ids:
    w = tokenizer.decode([id])
    tokens_na_string.append((idx, idx+len(w)))
    idx += len(w)

print(tokens_na_string)

# %%
start_end = []

for i, (a,b) in enumerate(tokens_na_string):
    sm = np.sum(resposta[a:b])
    if sm>1:
        start_end.append(i)

sentiment_ids = {'negative': 2430, 'positive': 1313, 'neutral': 7974}
 
# %%
#esse k é a coluna
s_id = sentiment_ids[train.loc[n, 'sentiment']]
input_ids[n, :len(enc.ids) + 2 + 2 + 1] = [0] + enc.ids + [2, 2] + [s_id] + [2]
attention_mask[n, :len(enc.ids) + 2 + 2 +1 ] = 1

# %%
tokenizer2 = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer2.save_vocabulary('.')

# %%
