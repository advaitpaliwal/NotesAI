import time
import nltk
from sentence_transformers import SentenceTransformer, util
from lexrank import STOPWORDS, LexRank
import numpy as np

def get_lexrank_summary(sentences, summary_size: int = 10, threshold: float = 0.1):
    lxr = LexRank(sentences, stopwords=None)
    summary = lxr.get_summary(sentences, summary_size=summary_size, threshold=threshold)
    return summary

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('transcript.txt', 'r') as f:
    corpus_sentences = nltk.sent_tokenize(f.read())
print(corpus_sentences)
print("Encode the corpus. This might take a while")
corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

print("Start clustering")
start_time = time.time()

clusters = util.community_detection(corpus_embeddings, min_community_size=3, threshold=0.5)

print("Clustering done after {:.2f} sec".format(time.time() - start_time))

with open('output.txt', 'w') as f:
    for i, cluster in enumerate(clusters):
        f.write(f"\n\nCluster {i+1}\n")
        cluster_text = ""
        for sentence_id in cluster:
            cluster_text += corpus_sentences[sentence_id] + " "
        summary = get_lexrank_summary( nltk.sent_tokenize(cluster_text))
        summary = '\n'.join(summary)
        f.write("\t" + summary)
