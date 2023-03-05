import streamlit as st
from corpus_grabber import CorpusGrabber
from text_segmenter import TextSegmenter
import time


def main():
    st.title("Notes AI")

    # Initialize CorpusGrabber and TextSegmenter
    cg = CorpusGrabber()
    ts = TextSegmenter()

    # Get URL input from user
    url = st.text_input("Enter YouTube URL:")

    # Get text from URL using CorpusGrabber
    if st.button("Get Text"):
        corpus = cg.get_text(url)
        notes = ts.segment_text(corpus, plot=True)
        st.write("Text segments:")
        # for i, note in enumerate(notes):
        #     st.write(f"{i + 1}. {note}")

        for note in notes:
            time.sleep(0.1)
            max_retries = 3
            flag = True
            for i in range(max_retries):
                n = ts.get_notes(note)
                try:
                    notes_json = ts.text_to_json(n)
                    heading = notes_json['heading']
                    st.write('# ' + heading)
                    bullets = notes_json['summary']
                    for bullet in bullets:
                        st.write('- ' + bullet)
                    flag = False
                    break
                except Exception as e:
                    print(e)
                    print(n)
            if flag:
                st.write('# Error getting this chapter')
            # st.write(ts.get_notes(note) + "\n\n===\n\n")
        
        st.write('===')

if __name__ == "__main__":
    main()
