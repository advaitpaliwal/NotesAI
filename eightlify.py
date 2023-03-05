import re

import requests


def extract_youtube_id(video_url):
    if "youtube" not in video_url:
        raise ValueError("Invalid YouTube URL")
    youtube_id = re.search(r'v=([^&]+)', video_url).group(1)
    return youtube_id.strip()


headers = {

    "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTY3NzkzMjQxMCwianRpIjoiMGE0YTY0NzctZTU2Zi00NmY0LWEzNTktZTk1YzI4NGE1YzljIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6ImFkdmFpdHNwYWxpd2FsQGdtYWlsLmNvbSIsIm5iZiI6MTY3NzkzMjQxMCwiZXhwIjoxNjgwNTI0NDEwfQ.69HLu1_YSmMFHAv_J7ihfLYeSNiJ-WrB9Vh8eYC78qk",
    "origin": "https://frontend.eightify.app",
    "referer": "https://frontend.eightify.app/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.63"}
video_url = "https://www.youtube.com/watch?v=bueBi2lOuAM"
data = {"refill": "100"}
url = f"https://backend.eightify.app/key-ideas?video_id={extract_youtube_id(video_url)}&language=EN&auto_summary=false&source=summarize-button"
url = "https://backend.eightify.app/balance"
response = requests.get(url, headers=headers)

print(response.text)
