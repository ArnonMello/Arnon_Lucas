
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
# %%
stop = set(stopwords.words('english'))
plt.style.use('seaborn')
# %%
#Abrindo os DataFrames
train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv")


# %%
#Checando informações sobre as colunas
train.info()

# %%
#Verificando os valores nulos
train[train['text'].isna()]
# %%
#Retirando os valores nulos
train.dropna(inplace = True) 
# %%
#Ajeitando a numeração das linhas
train.reset_index(inplace=True) 

# %%
train.tail()

# %%
train.info()

# %%
train['sentiment'].unique()

#%%
train['sentiment'] = train['sentiment'].astype('category')

# %%
train.info()


# %%
sns.countplot(x='sentiment', data=train)
plt.show()
sns.countplot(x='sentiment', data=test)

# %%
#Criando uma coluna com o tamanho de cada texto
train['chars'] = train['text'].apply(len)
train['selected_chars'] = train['selected_text'].apply(len)

# %%
g = sns.FacetGrid(train, hue = "sentiment")
g = g.map(sns.kdeplot, "chars").add_legend()
g.fig.suptitle('Distribuição do número de letras por tweet')


# %%
g = sns.FacetGrid(train, hue = "sentiment")
g = g.map(sns.kdeplot, "selected_chars").add_legend()
g.fig.suptitle('Distribuição do numero de letras por trecho extraído')
# %%
train.head()

# %%
train['word_count'] = train['text'].apply(lambda s: len(set(re.findall(r"\w+", s))))

# %%
train.head()


# %%
g = sns.FacetGrid(train, hue = 'sentiment', col = 'sentiment')
g = g.map(sns.distplot, 'word_count').add_legend()
g.fig.suptitle('Distribuição do número de palavras única por tweet')

# %%
def preprocess(df, stop = stop, n=1, col='text'):

    new_corpus=[]
    stem = PorterStemmer()
    lem = WordNetLemmatizer()
    for text in df[col]:
        words = [w for w in word_tokenize(text) if (w not in stop)]

        words = [lem.lemmatize(w) for w in words if (len(w)>n)]

        new_corpus.append(words)

    new_corpus = [word for l in new_corpus for word in l]

    return new_corpus

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'https?://www\.\S+\.com', '', texto)
    texto = re.sub(r'^[A-Za-z|\s]', '', texto)
    return texto

def limpar_df(df, cols):
    for col in cols:
        df[col] = df[col].astype(str).apply(lambda x: limpar_texto(x))
    return df
# %%
sents = ['negative', 'positive', 'neutral']
colors = ['r', 'g', 'b']
fig, ax = plt.subplots(1, 3)

for i in range(3):
    df = limpar_df(train[train.sentiment == sents[i]], ['selected_text'])
    corpus_train = preprocess(df, n=3, col = 'selected_text')
    counter = Counter(corpus_train)
    most = counter.most_common()
    x = []
    y = []
    for word,count in most[:20]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    sns.barplot(x=y, y=x, ax=ax[i], color = colors[i])
    ax[i].set_title(sents[i], color = colors[i])
fig.suptitle('Palavras mais comuns por sentimento')

