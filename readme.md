# NotesSpace
NotesSpace is a user-friendly web application designed for efficient note-taking. It enables you to input YouTube or file URLs and generates notes for you. The primary code for this application can be found in the NotesSpace folder, while the contents of the test folder were used for project experimentation.

# Key Files
The following are key files responsible for different functionalities of the NotesSpace app:

- `corpus_grabber.py`: This file grabs text from YouTube videos and files and preprocesses them.
- `text_segmentation.py`: This file semantically segments the text by topic.
- `app.py`: This file contains the Streamlit front end for the app.

# How to Run
To run NotesSpace on your local machine, follow these simple steps:

1. Download the repository onto your machine.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run `streamlit run NotesSpace/app.py` on your command line.
4. Enjoy!

5. We hope that this app will make note-taking much more manageable for you. If you encounter any issues or have suggestions for improvements, please let us know.