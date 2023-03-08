from PyPDF2 import PdfReader

# creating a pdf reader object
reader = PdfReader('thesis.pdf')

# printing number of pages in pdf file
corpus = ""
for i in range(len(reader.pages)):
    print(i)
    print()
    print(reader.pages[i].extract_text())
    corpus += reader.pages[i].extract_text()
    print()