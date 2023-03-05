import streamlit as st
from corpus_grabber import CorpusGrabber
from text_segmenter import TextSegmenter


def main():
    st.title("Notes AI")

    # Initialize CorpusGrabber and TextSegmenter
    cg = CorpusGrabber()
    ts = TextSegmenter()

    # Create placeholders for input components
    url_input_placeholder = st.empty()
    file_input_placeholder = st.empty()

    # Get input type from user
    input_type = st.selectbox("Select input type", ["URL", "File"])

    # Show appropriate input component based on selection
    if input_type == "URL":
        # Remove file input component
        file_input_placeholder.empty()

        # Show URL input component
        url = url_input_placeholder.text_input("Enter YouTube URL:")
        if st.button('Extract Topics'):
            corpus = cg.get_text(url)
            topics = ts.extract_topics(corpus)
            st.write(f"Topics extracted from {url}:")
            st.plotly_chart(ts.plot_topics(topics))
        if st.button("Get Text"):
            corpus = cg.get_text(url)
            notes = ts.segment_text(corpus, plot=True)
            st.write(f"Text segments extracted from {url}:")
            for i, note in enumerate(notes):
                st.write(f"{i + 1}. {note}")

    elif input_type == "File":
        # Remove URL input component
        url_input_placeholder.empty()

        # Show file input component
        uploaded_file = file_input_placeholder.file_uploader("Choose a file", type=["txt", "mp3", "mp4", "pptx", "pdf"],
                                                             accept_multiple_files=False)

        if uploaded_file:
            file_type = uploaded_file.type.split("/")[-1]
            st.write(f"Processing file: {uploaded_file.name} ({file_type})")

            if file_type == "txt":
                # Process text file
                corpus = uploaded_file.read()
                notes = ts.segment_text(corpus, plot=True)
                st.write(f"Text segments extracted from {uploaded_file.name}:")
                for i, note in enumerate(notes):
                    st.write(f"{i + 1}. {note}")

            elif file_type == "mp3":
                st.write("Audio processing coming soon!")
            elif file_type == "mp4":
                st.write("Video processing coming soon!")
            elif file_type == "pptx":
                st.write("PowerPoint processing coming soon!")
            elif file_type == "pdf":
                st.write("PDF processing coming soon!")


if __name__ == "__main__":
    main()
