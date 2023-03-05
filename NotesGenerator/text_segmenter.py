import re

from gensim.models import LdaMulticore
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import concurrent.futures
import gensim
from gensim import corpora
import plotly.express as px
import pandas as pd

import openai

openai.api_key = "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"


class TextSegmenter:
    def __init__(self, split_method="spacey"):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.split_method = split_method
        if split_method == "spacey":
            import spacy
            self.split_model = spacy.load("en_core_web_sm")
        elif split_method == "nnsplit":
            from nnsplit import NNSplit
            self.split_model = NNSplit.load("en")

    def text_to_json(self, json_string):
        match = re.search("({.*})", json_string)
        return eval(match.group(1))

    def get_notes(self, segment):
        prompt = f'''You are an intelligent chatbot that creates concise notes based on plain text. Your notes should include relevant and non vague information.

                    Here is a part of the text:
                    {segment}

                    Generate notes based on this segment. Your response should be in JSON format 
                    with heading as the key and a list of bullet points as value.
                    '''
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=256,
                n=1,
            )
            yield self.text_to_json(response["choices"][0]["text"].strip())

        except:
            try:
                response = openai.ChatCompletion.create(
                    model="davinci",
                    prompt=prompt,
                    max_tokens=256,
                    n=1,
                )
                yield self.text_to_json(response["choices"][0]["message"]["content"])
            except:
                raise Exception("Rate limit reached")

    def get_segments(self, corpus):
        global segments
        if self.split_method == "spacey":
            print(corpus)
            doc = self.split_model(corpus)
            segments = [sent.text for sent in doc.sents]
        elif self.split_method == "nnsplit":
            splits = self.split_model.split([corpus], )[0]
            segments = [str(sentence) for sentence in splits]
        filtered_segments = []
        for segment in segments:
            if len(segment.split()) > 20:
                filtered_segments.append(segment)
        return filtered_segments

    def get_embedding(self, text):
        embedding = self.embedding_model.encode(text)
        return embedding

    def get_embeddings(self, texts):
        embeddings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.get_embedding, text) for text in texts
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    embedding = future.result()
                except Exception as e:
                    print(f"Error: {e}")
                else:
                    embeddings.append(embedding)
        return embeddings

    def measure_similarity(self, embeddings):
        sim = []
        for i in range(len(embeddings)):
            comparison_index = i + 1
            if comparison_index < len(embeddings):
                sim.append(
                    cos_sim(embeddings[i], embeddings[comparison_index]))
            else:
                sim.append(0)
        return sim

    def find_peaks_in_similarity(self, sim):
        peaks, _ = find_peaks(np.asarray(sim[:-1]) * -1,
                              prominence=0.1)
        return peaks

    def plot_similarity(self, vid, peaks):
        plt.figure().set_figheight(2)
        plt.figure().set_figwidth(20)
        plt.plot(vid.index, vid["similarity"])
        plt.ylim((0, 1))
        plt.plot(peaks, vid.loc[peaks]["similarity"], "x")
        plt.show()

    def segment_text(self, corpus, plot=False):
        segments = self.get_segments(corpus)
        embeddings = self.get_embeddings(segments)
        sim = self.measure_similarity(embeddings)
        vid = pd.DataFrame(
            {"segment": segments, "embedding": embeddings, "similarity": sim}
        )
        peaks = self.find_peaks_in_similarity(sim)
        if plot:
            self.plot_similarity(vid, peaks)
        dfs = []
        last_check = 0
        for ind in peaks:
            dfs.append(vid.loc[last_check:ind - 1])
            last_check = ind
        segmented_by_similarity_change = [
            " ".join(df["segment"].values) for df in dfs
        ]
        return segmented_by_similarity_change
        # for segment in segmented_by_similarity_change:
        #     yield from self.get_notes(segment)

    def extract_topics(self, text, num_topics=5, num_words=5):
        text_data = text.split('.')
        # Tokenize the text data
        tokenized_text = [gensim.utils.simple_preprocess(sentence) for sentence in text_data]

        # Remove stop words
        stop_words = gensim.parsing.preprocessing.STOPWORDS
        tokenized_text = [[word for word in sentence if word not in stop_words] for sentence in tokenized_text]

        # Create a dictionary of the tokenized text data
        dictionary = corpora.Dictionary(tokenized_text)

        # Create a bag-of-words representation of the text data
        bow_corpus = [dictionary.doc2bow(text) for text in tokenized_text]

        # Train an LDA model on the bag-of-words corpus
        lda_model = LdaMulticore(bow_corpus, num_topics=2, id2word=dictionary, passes=10, workers=2)

        # Extract the major topics from the text data
        topics = lda_model.show_topics(num_topics=-1, formatted=False)
        return topics

    def plot_topics(self, topics):
        # create a list of all the words in the topics
        for topic in topics:
            df = pd.DataFrame(topic[1], columns=['word', 'freq'])
            fig = px.scatter(df, x='word', y='freq', size='freq', title=f'Topic {topic[0]}')
            return fig
