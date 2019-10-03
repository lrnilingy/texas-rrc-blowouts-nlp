import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

stemmer = SnowballStemmer('english')


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 1:
            result.append(lemmatize_stemming(token))
    return result


data = pd.read_csv('../data/blowouts.csv')
docs = data[['Remarks']]
processed_docs = docs['Remarks'].fillna('').astype(str).map(preprocess)

# plot a simple word cloud
wordcloud = WordCloud(width=1600, height=800).generate(processed_docs.to_string())
plt.figure(figsize=(20, 10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
