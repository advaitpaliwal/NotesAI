import re

from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptAnalyzer:
    def __init__(self, video_url):
        self.transcript = self.get_text_timed(self.extract_youtube_id(video_url))

    def extract_youtube_id(self, video_url):
        if "youtube" not in video_url:
            raise ValueError("Invalid YouTube URL")
        youtube_id = re.search(r'v=([^&]+)', video_url).group(1)
        return youtube_id.strip()

    def get_text_timed(self, youtube_id):
        try:
            return YouTubeTranscriptApi.get_transcript(youtube_id, languages=['en', 'en-GB', 'en-US'])
        except:
            raise ValueError("Unable to transcribe YouTube URL")

    def timed_to_string(self):
        return ' '.join([t['text'] for t in self.transcript])


youtube_url = "https://www.youtube.com/watch?v=IDDmrzzB14M&t=45s&ab_channel=CS50"

yt = YouTubeTranscriptAnalyzer(youtube_url)
with open('transcript.txt', 'w') as f:
    f.write(yt.timed_to_string())
