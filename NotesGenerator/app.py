import streamlit as st
from corpus_grabber import CorpusGrabber
from text_segmenter import TextSegmenter
import time
from streamlit_player import st_player


def main():
    # st.title("Notes AI")
    st.image('logo.png')
    st.write('''
            # How to use
            (1) Choose where you want to add text from\n
            (2) Upload file or add link (direct link to youtube video, pptx or pdf file)\n
            (2) Use "Extract Topics" and "Get Notes" Buttons to get notes and topics!\n
            That's it :)\n\n\n
            ### Now Go ahead:
        ''')
    notes = None

    # Initialize CorpusGrabber and TextSegmenter
    cg = CorpusGrabber()
    ts = TextSegmenter()

    # Get input type from user
    input_type = st.selectbox("(1) Select input type:", ["URL", "File"])

    # Create placeholders for input components
    url_input_placeholder = st.empty()
    file_input_placeholder = st.empty()


    # Show appropriate input component based on selection
    if input_type == "URL":
        # Remove file input component
        file_input_placeholder.empty()

        # Show URL input component
        url = url_input_placeholder.text_input("(2) Enter URL for youtube video or a URL that ends with .pptx/.pdf:")
        if st.button('Extract Topics'):
            st.write('Getting topics, please be *very* patient :)')
            corpus = cg.get_text(url)
            st.write('Dont worry, still getting them...')
            topics = ts.extract_topics(corpus)
            st.write(f"Topics extracted from {url}:")
            st.plotly_chart(ts.plot_topics(topics))
        if st.button("Get Notes"):
            st_player(url)
            st.write('Getting notes, please be *very* patient :)')
            corpus = cg.get_text(url)
            st.write('Dont worry, still getting them...')
            notes = ts.segment_text(corpus, plot=True)
            st.write(f"Text segments extracted from {url}:")

    elif input_type == "File":
        # Remove URL input component
        url_input_placeholder.empty()

        # Show file input component
        uploaded_file = file_input_placeholder.file_uploader("(2) Choose a file:", type=["txt", "mp3", "mp4", "pptx", "pdf"],
                                                             accept_multiple_files=False)

        if uploaded_file:
            file_type = uploaded_file.type.split("/")[-1]
            st.write(f"Processing file: {uploaded_file.name} ({file_type})")

            if file_type == "txt":
                # Process text file
                corpus = uploaded_file.read()
                notes = ts.segment_text(corpus, plot=True)
                st.write(f"Text segments extracted from {uploaded_file.name}:")
                # for i, note in enumerate(notes):
                #     st.write(f"{i + 1}. {note}")
            elif file_type == "mp3":
                st.write("Audio processing coming soon!")
            elif file_type == "mp4":
                st.write("Video processing coming soon!")
            elif file_type == "pptx":
                st.write("PowerPoint processing coming soon!")
            elif file_type == "pdf":
                st.write("PDF processing coming soon!")

    if notes is not None:
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
        
        st.write("# ('U') === End of Chapters === ('U')")
            


if __name__ == "__main__":
    main()
