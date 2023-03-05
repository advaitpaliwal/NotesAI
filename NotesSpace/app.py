from datetime import datetime
import streamlit as st
from corpus_grabber import CorpusGrabber
from text_segmenter import TextSegmenter
import time


def get_total_words(history):
    total_words = 0
    for item in history:
        total_words += item['word_count']
    return total_words


def get_total_time_saved(history):
    total_time_saved = 0
    for item in history:
        total_time_saved += item['time_saved']
    return total_time_saved


def statistics_page(cg=None, ts=None):
    # Total words count
    total_words = get_total_words(st.session_state.history)
    st.subheader(f"Total words processed: `{total_words}`")

    # Total time saved
    total_time_saved = get_total_time_saved(st.session_state.history)
    st.subheader(f"Total time saved: `{total_time_saved}` minutes")

    # History list
    st.subheader("Notes history:")
    for item in st.session_state.history:
        st.write(item['url'])
        st.download_button(
            label="Download Notes",
            data=item['notes'],
            file_name=item['file_name'],
            mime="text/plain",
        )


def generate_notes_page(cg, ts):
    # Initialize history list if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []

    url = st.text_input("Enter URL for youtube video or a URL that ends with .pdf:")
    button = st.button("Get Notes")
    col1, col2 = st.columns([2, 2])

    if url and button:
        with col1:
            st.subheader("Preview Source Material")
            if url.endswith('.pdf') or url.endswith('.pptx'):
                pdf_url = f"https://drive.google.com/viewerng/viewer?embedded=true&url={url}"
                st.markdown(f'<iframe src="{pdf_url}" width="700" height="1000" frameborder="0"></iframe>',
                            unsafe_allow_html=True)
            else:
                st.video(url)

        with col2:
            if url:
                with st.spinner("Getting Notes... "):
                    corpus = cg.get_text(url)
                    word_count = len(corpus.split())
                    notes = ts.segment_text(corpus, plot=True)
                    if notes is not None:
                        notes_text = ""
                        topics = ts.extract_topics(corpus)
                        st.success(f"Topics and notes extracted from {url}:")
                        st.plotly_chart(ts.plot_topics(topics))
                        st.success(f"Processed **{word_count}** words")
                        for note in notes:
                            time.sleep(0.1)
                            max_retries = 3
                            flag = True
                            for i in range(max_retries):
                                n = ts.get_notes(note)
                                try:
                                    notes_json = ts.text_to_json(n)
                                    heading = notes_json['heading']
                                    notes_text += '\n## ' + heading + '\n'
                                    st.write('## ' + heading)
                                    bullets = notes_json['summary']
                                    for bullet in bullets:
                                        notes_text += '- ' + bullet + '\n'
                                        st.write('- ' + bullet)
                                    flag = False
                                    break
                                except Exception as e:
                                    print(e)
                                    print(n)
                            if flag:
                                print('# Error getting this chapter')

                        st.session_state.history.append(
                            {'url': url, 'word_count': word_count, 'time_saved': int(word_count / 200),
                             'notes': notes_text,
                             "file_name": f"notes-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"})
                    # Show end of chapters separator in the notes section
                    st.write("# ('U') === End of Chapters === ('U')")


def main():
    st.set_page_config(layout='wide')
    st.image('NotesSpace/logo.png')
    # Main content
    cg = CorpusGrabber()
    ts = TextSegmenter()

    pages = {
        "Generate Notes": generate_notes_page,
        "History": statistics_page,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]
    page(cg, ts)


if __name__ == "__main__":
    main()
