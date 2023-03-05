import streamlit as st
from corpus_grabber import CorpusGrabber
from text_segmenter import TextSegmenter


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
        notes = ts.segment_text(corpus)
        st.write("Text segments:")
        for i, note in enumerate(notes):
            st.write(f"{i + 1}. {note}")


if __name__ == "__main__":
    main()
