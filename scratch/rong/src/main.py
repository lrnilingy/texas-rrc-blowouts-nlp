import pandas as pd
import gensim
from gensim import corpora
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


data = pd.read_csv('../data/cleaned_values.csv')
docs = data[['Remarks']]
processed_docs = docs['Remarks'].fillna('').astype(str).map(preprocess)

# 1. plot a simple word cloud
wordcloud = WordCloud(width=1600, height=800, max_words=50).generate(processed_docs.to_string())
plt.figure(figsize=(20, 10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()
plt.savefig('../output/simple_wordcloud.png', facecolor='k', bbox_inches='tight')


# 2. topic modeling
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(text) for text in processed_docs]

NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=150)

topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
