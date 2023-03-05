import gensim
from gensim import corpora
from gensim.models.lsimodel import LsiModel
import plotly.express as px
import pandas as pd

with open('transcript.txt', 'r') as file:
    text_data = file.read()

#segment text into sentences
text_data = text_data.split('.')
print(text_data)
# Tokenize the text data
tokenized_text = [gensim.utils.simple_preprocess(sentence) for sentence in text_data]

# Remove stop words
stop_words = gensim.parsing.preprocessing.STOPWORDS
tokenized_text = [[word for word in sentence if word not in stop_words] for sentence in tokenized_text]

# Create a dictionary of the tokenized text data
dictionary = corpora.Dictionary(tokenized_text)

# Create a bag-of-words representation of the text data
bow_corpus = [dictionary.doc2bow(text) for text in tokenized_text]

# Train an LSI model on the bag-of-words corpus
lsi_model = LsiModel(bow_corpus, num_topics=2, id2word=dictionary)

# Extract the major topics from the text data
topics = lsi_model.show_topics(num_topics=-1, formatted=False)

# Create a plotly scatter plot for each topic
for topic in topics:
    df = pd.DataFrame(topic[1], columns=['word', 'freq'])
    fig = px.scatter(df, x='word', y='freq', size='freq', title=f'Topic {topic[0]}')
    fig.show()
