import streamlit as st
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from chatbot import get_best_answer

# Load preprocessed data, embeddings, and model
def load_chatbot():
    data = pd.read_excel('nhs_dataset.xlsx')
    with open('column_embeddings.pkl', 'rb') as f:
        column_embeddings = pickle.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    return data, column_embeddings, model

# Streamlit UI for chatbot interaction
def main():
    st.set_page_config(page_title="NHS Wales Chatbot", page_icon="🏥", layout="centered")
    st.title('NHS Wales Organizational Map Chatbot')

    st.write("""
        Welcome! This chatbot helps you learn about the structure and roles of NHS bodies in Wales.
        Ask about organizations, their responsibilities, reporting lines, and more.
    """)

    user_input = st.text_input("📝 Enter your question:")

    if user_input:
        data, column_embeddings, model = load_chatbot()

        # Get the best answer based on cosine similarity
        entity, column, response = get_best_answer(user_input, column_embeddings, model, data)

        if entity and column and response and "sorry" not in response.lower():
            st.success("✅ Match found!")
            st.write(f"**Health Body:** {entity}")
            st.write(f"**Column Matched:** {column}")
            st.write(f"**Answer:** {response}")
        else:
            st.warning("⚠️ Sorry, I couldn't find a confident match.")
            st.write("Try rephrasing your question or being more specific.")
    
    # Footer in Streamlit
    st.markdown(
        """
        <div style="text-align: center; font-size: 12px; color: #888888;">
            <p><em>Disclaimer:</em> This chatbot is designed to provide information about the NHS Wales hierarchy. While every effort is made to provide accurate responses, please note that the information provided is based on publicly available data and internet research. The chatbot may occasionally make mistakes, and some information might not be up to date or fully accurate. Please verify critical details with official NHS Wales resources.
            </p>
        </div>
        """, unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
