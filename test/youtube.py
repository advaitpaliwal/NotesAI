import re

from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptAnalyzer:
    def __init__(self, video_url):
        self.youtube_id = self.extract_youtube_id(video_url)
        self.transcript = self.get_text_timed()

    @staticmethod
    def extract_youtube_id(video_url):
        if "youtube" not in video_url:
            raise ValueError("Invalid YouTube URL")
        youtube_id = re.search(r'v=([^&]+)', video_url).group(1)
        return youtube_id.strip()

    def get_text_timed(self):
        try:
            return YouTubeTranscriptApi.get_transcript(self.youtube_id, languages=['en', 'en-GB', 'en-US'])
        except:
            raise ValueError("Unable to transcribe YouTube URL")

    def timed_to_string(self):
        return ' '.join([t['text'] for t in self.transcript])


youtube_url = "https://www.youtube.com/watch?v=EQISygAle3M"

yt = YouTubeTranscriptAnalyzer(youtube_url)
timed_text = yt.get_text_timed()
l = [0,

     194,

     404,

     528,

     661,

     766,

     889,

     1083]
result = []
j = 1
text = ""
for i in range(len(timed_text)):
    if timed_text[i]["start"] <= l[j]:
        text += timed_text[i]["text"] + " "
    else:
        result.append(text)
        text = ""
        j += 1
        if j >= len(l):
            break

print(result)