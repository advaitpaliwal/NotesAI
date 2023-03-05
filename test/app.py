import streamlit as st

# Define the main function that runs the app
def main():
    st.title("Notes AI")
    st.write("Convert your audio, video, and document files to educational notes.")

    # Create a file uploader and ask the user to upload a file
    file = st.file_uploader("Upload a file", type=["txt", "mp3", "mp4", "pdf", "pptx"])

    # If a file is uploaded, show a confirmation message and offer to convert it
    if file:
        st.success("File uploaded successfully!")
        if st.button("Convert"):
            st.progress("Converting file...")

    # If no file is uploaded, show a message prompting the user to upload a file
    else:
        st.warning("Please upload a file.")

if __name__ == "__main__":
    main()
