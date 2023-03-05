import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Load the transcript text file
with open('transcript.txt', 'r') as file:
    text = file.read()

# Preprocess the text by tokenizing and removing stop words
tokenized_text = gensim.utils.simple_preprocess(text, deacc=True, min_len=2, max_len=15)
stop_words = gensim.parsing.preprocessing.STOPWORDS
tokenized_text = [word for word in tokenized_text if word not in stop_words]

# Create a dictionary from the tokenized text
dictionary = corpora.Dictionary([tokenized_text])

# Convert tokenized text to vectors
corpus = [dictionary.doc2bow([word]) for word in tokenized_text]

# Train the LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=3,
                     passes=10,
                     alpha='auto',
                     eta='auto')

# Print the topics and their corresponding words
for topic in lda_model.print_topics():
    print(f"Topic {topic[0]}: {topic[1]}")

# Segment the text based on topics
topics = lda_model.get_document_topics(corpus)
for i, doc in enumerate(topics):
    print(f"Document {i+1}: Topic {doc[0][0]}")
    print(text.split('.')[i])
    print("\n")
