import nltk
from nltk.tokenize import texttiling
from PyPDF2 import PdfReader

# creating a pdf reader object
reader = PdfReader('thesis.pdf')

# printing number of pages in pdf file
corpus = ""
for i in range(len(reader.pages)):
    corpus += reader.pages[i].extract_text()

# Tokenize the text using TextTiling
tokenizer = texttiling.TextTilingTokenizer()
segments = tokenizer.tokenize(corpus)

# Print the segments
for i, segment in enumerate(segments):
    print(f"Segment {i+1}:")
    print("--" * 40)
    print(segment)
    print("--" * 40)
