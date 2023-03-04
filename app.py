import streamlit as st

st.set_page_config(page_title="Notes AI", page_icon="✅", layout="wide")


def show_sidebar():
    with st.sidebar:
        st.header("**About**")
        st.write(
            "**GPT:green[Check]** is a plagiarism detection tool that helps you check the originality of your text. It generates text completions based on a given prompt and compares them to your answer. The similarity between the two texts is calculated using a weighted formula that combines Jaccard and Cosine similarity.")
        st.subheader("**What is n?**")
        st.write(
            "n is the number of different text completions generated by the AI model. Higher values may result in more diverse responses.")
        st.subheader("**What is cosine similarity?**")
        st.write("Cosine similarity is a measure of how similar two pieces of text are based on semantics.")
        st.subheader("**What is jaccard similarity?**")
        st.write(
            "Jaccard similarity is a measure of how similar two pieces of text are based on number of common words.")


def get_user_input():
    prompt = st.text_area("**Enter the prompt:**", max_chars=1000)
    student_answer = st.text_area("**Enter the answer:**", height=200)
    n = st.slider("**n:**", 1, 5, 3, 1)
    return prompt, student_answer, n


def check_plagiarism(prompt, student_answer, n):
    if prompt == "":
        return st.warning("Please enter a prompt.")
    if student_answer == "":
        return st.warning("Please enter an answer.")
    if len(prompt) < 10:
        return st.warning("Please enter a prompt with at least 10 characters.")
    if len(student_answer) < 250:
        return st.warning("Please enter an answer with at least 250 characters.")
    with st.spinner("Initializing modules..."):
        if not st.session_state.get("detector"):
            st.session_state.detector = PlagiarismDetector()
    with st.spinner("Processing request..."):
        try:
            generated_answers = st.session_state.detector.generate_answers(prompt, n)
        except Exception as e:
            st.error(e)
            st.stop()
        results = st.session_state.detector.check_plagiarism(generated_answers, student_answer)
        st.header("Similarity Results:")
        i = 1
        avg_overall_similarity = 0
        for answer, similarity in results.items():
            jaccard_similarity = similarity['jaccard']
            cosine_similarity = similarity['cosine']
            overall_similarity = similarity['overall']
            with st.expander(f"{round(overall_similarity * 100, 2)}%"):
                st.write("**Cosine:**", f"`{round(cosine_similarity * 100, 2)}%`")
                st.write("**Jaccard:**", f"`{round(jaccard_similarity * 100, 2)}%`")
                st.markdown(answer)
                i += 1
                avg_overall_similarity += overall_similarity
        avg_overall_similarity /= len(results)
        st.info(f"Average similarity: {round(avg_overall_similarity * 100, 2)}%")
        if avg_overall_similarity > 0.5:
            st.warning("Your answer is plagiarized!")
        else:
            st.success("Your answer is original!")


def main():
    st.title("GPT:green[Check]")
    show_sidebar()
    prompt, student_answer, n = get_user_input()
    if st.button("Detect"):
        check_plagiarism(prompt, student_answer, n)


if __name__ == "__main__":
    main()
