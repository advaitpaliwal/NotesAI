import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from nnsplit import NNSplit

openai.api_key = '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' # replace with openai api key

def segment_video(video_id: str, languages: list[str]) -> list[str]:
    """
    Segments a YouTube video into sections based on changes in the transcript's embedding scores.

    Args:
        video_id: The ID of the YouTube video to segment.
        languages: A list of language codes to use for the transcript.

    Returns:
        A list of strings representing the segmented video.
    """
    try:
        # Load video transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id.strip(), languages=languages)
        corpus = ' '.join([t['text'] for t in transcript])

        # Split corpus into segments
        splitter = NNSplit.load("en")
        splits = [str(sentence) for sentence in splitter.split([corpus])[0]]

        # Get GPT embeddings for each segment
        seg_embeddings = [get_embedding(seg) for seg in splits]

        # Measure similarity between each embedding and the next in time series
        sim = [cosine_similarity(seg_embeddings[i], seg_embeddings[i+1]) if i < len(seg_embeddings)-1 else 0 for i in range(len(seg_embeddings))]

        # Make pandas dataframe for all data
        vid = pd.DataFrame({'segment': splits, 'embedding': seg_embeddings, 'similarity': sim})

        # Find moments of big change in video
        peaks, _ = find_peaks(np.asarray(list(vid['similarity'].values)[:-1])*-1, prominence=0.1) # sensitivity can be tuned with prominence arg
        peaks_locs = vid.iloc[peaks]

        # Uncomment following code to see how video is segmented
        # plt.figure().set_figheight(2)
        # plt.figure().set_figwidth(20)
        # plt.plot(vid.index, vid['similarity'])
        # plt.ylim((0,1))
        # plt.plot(peaks, peaks_locs['similarity'], "x")
        # plt.show()

        # Break up video transcript based on change in embedding score
        dfs = []
        last_check = 0
        for ind in peaks:
            dfs.append(vid.loc[last_check:ind-1])
            last_check = ind
        segmented_by_similarity_change = [' '.join(df['segment'].values) for df in dfs]

        return segmented_by_similarity_change

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
