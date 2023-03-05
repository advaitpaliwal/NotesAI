from NotesGenerator.TextSegmentation import TextSegmenter
from NotesGenerator.CorpusGrabber import CorpusGrabber

ts = TextSegmenter()
cg = CorpusGrabber()

# corpus = cg.get_text('https://evolution.berkeley.edu/wp-content/uploads/2022/02/Evo101_03_Mechanisms_UE.pdf') # pdf test
# corpus = cg.get_text('https://www.rimsd.k12.ca.us/cms/lib/CA02206080/Centricity/Domain/177/NaturalSelectionNotesPowerPoint.pptx') # pptx test
corpus = cg.get_text('https://www.youtube.com/watch?v=SV9AYYytbVM')
# print(corpus)
print(ts.segment_text(corpus, plot=True))