from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

with open('transcript.txt', 'r') as f:
    corpus = f.readlines()

# Tokenize the corpus and create a bag-of-words representation
stop_words = set(stopwords.words('english') + list(string.punctuation))
texts = [[word for word in word_tokenize(document.lower()) if word not in stop_words] for document in corpus]

# Create a dictionary and corpus
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


# Train an LDA model with 2 topics
num_topics = 10
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Print the top 3 most significant words for each topic
for i, topic in lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False):
    print('Topic {}: {}'.format(i, ' '.join([word for word, _ in topic])))

# Segment a new document into topics
new_doc = 'I love my cute hamster but I also like to eat spinach.'
new_doc_bow = dictionary.doc2bow(new_doc.lower().split())
topic_probs = lda_model.get_document_topics(new_doc_bow)
most_likely_topic, _ = max(topic_probs, key=lambda x: x[1])
print('Most likely topic for the new document is {}'.format(most_likely_topic))
