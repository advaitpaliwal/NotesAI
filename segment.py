import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import spacy
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import concurrent.futures

def get_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # encode the text using SentenceTransformer
    embedding = model.encode(text)
    return embedding

def get_embeddings(texts):
    embeddings = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_embedding, text) for text in texts]
        for future in concurrent.futures.as_completed(futures):
            try:
                embedding = future.result()
            except Exception as e:
                print(f"Error: {e}")
            else:
                embeddings.append(embedding)
    return embeddings

def segment_video():
    # load video
    transcript = YouTubeTranscriptApi.get_transcript(str('EQISygAle3M').strip(), languages=['en', 'en-GB', 'en-US'])
    corpus = ' '.join([t['text'] for t in transcript])

    # splitting corpus into segments
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(corpus)
    segments = [sent.text for sent in doc.sents]
    # gets gpt embeddings for each segment
    seg_embeddings = get_embeddings(segments)
    # measuring similarity between each embedding and the next in timeseries
    sim = []
    for i in range(len(seg_embeddings)):
        comparison_index = i + 1
        if comparison_index < len(seg_embeddings):
            sim.append(cos_sim(seg_embeddings[i], seg_embeddings[comparison_index]))
        else:
            sim.append(0)

    # make pd df for all data
    vid = pd.DataFrame({'segment': segments, 'embedding': seg_embeddings, 'similarity': sim})

    # finding moments of big change in video
    peaks, _ = find_peaks(np.asarray(list(vid['similarity'].values)[:-1]) * -1,
                          prominence=0.1)  # sensitivity can be tuned with prominence arg
    peaks_locs = vid.iloc[peaks]
    # in order to see how video is segmented, uncomment following code
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
    segmented_by_smiliarity_change = [' '.join(df['segment'].values) for df in dfs]

    return segmented_by_smiliarity_change


if __name__ == '__main__':
    print(segment_video())
