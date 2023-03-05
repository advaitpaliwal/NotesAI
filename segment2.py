import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

def segment_video():
    # load video
    transcript = YouTubeTranscriptApi.get_transcript(str('IDDmrzzB14M').strip(), languages=['en', 'en-GB', 'en-US'])
    corpus = ' '.join([t['text'] for t in transcript])

    # splitting corpus into segments
    nltk.download('punkt')
    segments = nltk.sent_tokenize(corpus)

    # encode all segments using SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(segments)

    # calculate pairwise similarities between embeddings using matrix multiplication
    sim = np.matmul(embeddings, embeddings.T)

    # finding moments of big change in video
    sim_flat = np.concatenate(sim[:-1])
    peaks, _ = find_peaks(-sim_flat, prominence=0.1)
    peaks_locs = pd.DataFrame({'segment': segments, 'embedding': embeddings, 'similarity': sim[:,0]}).iloc[peaks]

    # in order to see how video is segmented, uncomment following code
    plt.figure().set_figheight(2)
    plt.figure().set_figwidth(20)
    plt.plot(peaks, peaks_locs['similarity'], "x")
    plt.show()

    # breaking up video transcript based on change in embedding score
    dfs = []
    last_check = 0
    for ind in peaks:
        dfs.append(segments[last_check:ind])
        last_check = ind
    segmented_by_smiliarity_change = [' '.join(df) for df in dfs]

    return segmented_by_smiliarity_change


if __name__ == '__main__':
    print(segment_video())
