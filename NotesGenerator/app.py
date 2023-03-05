import streamlit as st
from corpus_grabber import CorpusGrabber
from text_segmenter import TextSegmenter
import time
from io import BytesIO
import requests
from pptx import Presentation
from PIL import Image

def main():
    st.set_page_config(layout='wide')
    st.image('NotesGenerator/logo.png')

    # Sidebar
    st.sidebar.title("How to use")
    st.sidebar.markdown("- Choose where you want to add text from")
    st.sidebar.markdown("- Add link (direct link to youtube video, pptx or pdf file)")
    st.sidebar.markdown("- Use 'Extract Topics' and 'Get Notes' Buttons to get notes and topics!")

    # Main content
    notes = None

    cg = CorpusGrabber()
    ts = TextSegmenter()

    col1, col2 = st.columns([2, 2])
    url = st.text_input("Enter URL for youtube video or a URL that ends with .pdf/.pptx:")
    if st.button("Get Notes"):
        with col1:
            st.subheader("Preview Source Material")
            if url.endswith('.pdf'):
                pdf_url = f"https://drive.google.com/viewerng/viewer?embedded=true&url={url}"
                st.markdown(f'<iframe src="{pdf_url}" width="700" height="950" frameborder="0"></iframe>',
                            unsafe_allow_html=True)
            elif url.endswith('.pptx'):
                # Convert pptx to a sequence of images
                prs = Presentation(requests.get(url).content)
                slides = prs.slides
                images = []
                for slide in slides:
                    img = slide.background.background_style.fill_foreground_picture
                    if img:
                        fp = BytesIO()
                        img.save(fp, 'png')
                        images.append(fp.getvalue())

                # Display the images in the Streamlit app
                for image in images:
                    st.image(Image.open(BytesIO(image)))
            else:
                st.video(url)

        with col2:
            if url:
                st.write('Getting notes, please be *very* patient :)')
                st.write('Dont worry, still getting them...')
                corpus = cg.get_text(url)
                notes = ts.segment_text(corpus, plot=True)
                topics = ts.extract_topics(corpus)
                st.write(f"Topics extracted from {url}:")
                st.plotly_chart(ts.plot_topics(topics))
                st.write(f"Notes extracted from {url}:")

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
