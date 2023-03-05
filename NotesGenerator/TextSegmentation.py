

import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import concurrent.futures


class TextSegmenter():

    def __init__(self, split_method='spacey'):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.split_method = split_method
        if split_method == 'spacey':
            import spacy
            nlp = spacy.load('en_core_web_sm')
            self.split_model = nlp
        elif split_method == 'nnsplit':
            from nnsplit import NNSplit # https://bminixhofer.github.io/nnsplit/
            self.split_model = NNSplit.load("en")

    def get_segments(self, corpus):
        if self.split_method == 'spacey': 
            doc = self.split_model(corpus)
            segments = [sent.text for sent in doc.sents]
        elif self.split_method == 'nnsplit':
            splits = self.split_model.split([corpus], )[0]
            segments = [str(sentence) for sentence in splits]
        return segments

    def get_embedding(self, text):
        # encode the text using SentenceTransformer
        embedding = self.embedding_model.encode(text)
        return embedding

    def get_embeddings(self, texts):
        embeddings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_embedding, text)
                       for text in texts]
            for future in concurrent.futures.as_completed(futures):
                try:
                    embedding = future.result()
                except Exception as e:
                    print(f"Error: {e}")
                else:
                    embeddings.append(embedding)
        return embeddings

    def segment_text(self, corpus, plot=False):

        # splitting corpus into segments
        segments = self.get_segments(corpus)

        # gets gpt embeddings for each segment
        seg_embeddings = self.get_embeddings(segments)
        # measuring similarity between each embedding and the next in timeseries
        sim = []
        for i in range(len(seg_embeddings)):
            comparison_index = i + 1
            if comparison_index < len(seg_embeddings):
                sim.append(
                    cos_sim(seg_embeddings[i], seg_embeddings[comparison_index]))
            else:
                sim.append(0)

        # make pd df for all data
        vid = pd.DataFrame(
            {'segment': segments, 'embedding': seg_embeddings, 'similarity': sim})

        # finding moments of big change in video
        peaks, _ = find_peaks(np.asarray(list(vid['similarity'].values)[:-1]) * -1,
                              prominence=0.1, width=2)  # sensitivity can be tuned with prominence arg, change size tuned with width arg
        peaks_locs = vid.iloc[peaks]
        # in order to see how video is segmented, uncomment following code
        if plot:
            plt.figure().set_figheight(2)
            plt.figure().set_figwidth(20)
            plt.plot(vid.index, vid['similarity'])
            plt.ylim((0, 1))
            plt.plot(peaks, peaks_locs['similarity'], "x")
            plt.show()

        # breaking up video transcript based on change in embedding score
        dfs = []
        last_check = 0
        for ind in peaks:
            dfs.append(vid.loc[last_check:ind - 1])
            last_check = ind
        segmented_by_smiliarity_change = [
            ' '.join(df['segment'].values) for df in dfs]

        return segmented_by_smiliarity_change, vid
