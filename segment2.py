import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from scipy.signal import find_peaks
import spacy
import numpy as np
import matplotlib.pyplot as plt
from gensim.similarities import WmdSimilarity
from sentence_transformers import SentenceTransformer


def segment_video(video_id):
    # load video transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-GB', 'en-US'])
    corpus = ' '.join([t['text'] for t in transcript])

    # segmenting transcript into smaller segments
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(corpus)
    segments = [sent.text for sent in doc.sents]

    # generating embeddings for each segment
    model = SentenceTransformer('all-MiniLM-L6-v2')
    segment_embeddings = model.encode(segments)

    # computing similarity between segments
    wmd_instance = WmdSimilarity(segments, model)
    similarity = []
    for i in range(len(segments) - 1):
        similarity.append(wmd_instance[segments[i]], [segments[i+1]])

    # finding moments of significant change in video
    peaks, _ = find_peaks(np.asarray(similarity) * -1, prominence=0.1)
    peaks_locs = pd.DataFrame({'segment': [segments[i] for i in peaks], 'embedding': [segment_embeddings[i] for i in peaks],
                               'similarity': [similarity[i] for i in peaks]})

    # visualize segmentation
    plt.figure().set_figheight(2)
    plt.figure().set_figwidth(20)
    plt.plot(similarity)
    plt.ylim((0, 1))
    plt.plot(peaks, peaks_locs['similarity'], "x")
    plt.show()

    # segmenting transcript based on similarity change
    segmented_transcript = []
    last_check = 0
    for ind in peaks:
        segmented_transcript.append(' '.join(segments[last_check:ind]))
        last_check = ind
    segmented_transcript.append(' '.join(segments[last_check:]))

    return segmented_transcript
