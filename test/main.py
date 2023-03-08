from NotesSpace.text_segmenter import TextSegmenter
from NotesSpace.corpus_grabber import CorpusGrabber

ts = TextSegmenter()
cg = CorpusGrabber()

# corpus = cg.get_text('https://evolution.berkeley.edu/wp-content/uploads/2022/02/Evo101_03_Mechanisms_UE.pdf') # pdf test
# corpus = cg.get_text('https://www.rimsd.k12.ca.us/cms/lib/CA02206080/Centricity/Domain/177/NaturalSelectionNotesPowerPoint.pptx') # pptx test
corpus = cg.get_text('https://www.youtube.com/watch?v=TM6bt0XfodA&ab_channel=wattles')
with open('transcript.txt', 'w') as file:
    file.write(corpus)
# print(corpus)
s = ts.segment_text(corpus, plot=True)