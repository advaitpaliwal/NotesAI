import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import openai
openai.api_key = '' # replace with openai api key
from openai.embeddings_utils import get_embedding, cosine_similarity
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from nnsplit import NNSplit

"""
To install required libraries in python 3.9.16 run the following command:

    !python3.9 -m pip install nnsplit pandas youtube-transcript-api openai matplotlib plotly scipy sklearn scikit-learn

make sure to check that it is indeed installed onn 3.9.16
"""

def segment_video():
    
    # load video
    transcript = YouTubeTranscriptApi.get_transcript(str('flaSZRR2lqc').strip(), languages=['en', 'en-GB', 'en-US'])
    corpus = ' '.join([t['text'] for t in transcript])

    # splitting corpus into segments
    splitter = NNSplit.load("en")
    splits = splitter.split([corpus], )[0]
    segments = []
    for sentence in splits:
        segments.append(str(sentence))

    # gets gpt embeddings for each segment
    seg_embeddings = [get_embedding(seg) for seg in segments]

    # measuring similarity between each embedding and the next in timeseries
    sim = []
    for i in range(len(seg_embeddings)):
        comparison_index = i + 1
        if comparison_index < len(seg_embeddings):
            sim.append(cosine_similarity(seg_embeddings[i], seg_embeddings[comparison_index]))
        else:
            sim.append(0)

    # make pd df for all data
    vid = pd.DataFrame({'segment': segments, 'embedding': seg_embeddings, 'similarity': sim})

    # finding moments of big change in video
    peaks, _ = find_peaks(np.asarray(list(vid['similarity'].values)[:-1])*-1, prominence=0.1) # sensitivity can be tuned with prominence arg
    peaks_locs = vid.iloc[peaks]
    # in order to see how video is segmented, uncomment following code
    # plt.figure().set_figheight(2)
    # plt.figure().set_figwidth(20)
    # plt.plot(vid.index, vid['similarity'])
    # plt.ylim((0,1))
    # plt.plot(peaks, peaks_locs['similarity'], "x")
    # plt.show()

    # breaking up video transcript based on change in embedding score
    dfs = []
    last_check=0
    for ind in peaks:
        dfs.append(vid.loc[last_check:ind-1])
        last_check = ind
    segmented_by_smiliarity_change = [' '.join(df['segment'].values) for df in dfs]

    return segmented_by_smiliarity_change