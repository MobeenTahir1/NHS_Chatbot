import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the dataset and create embeddings for each relevant column
def load_data():
    data = pd.read_excel('nhs_dataset.xlsx')

    return data[['Health Body', 'Who Are They?', 'What Do They Do?', 'Established On What Basis?', 'Who Do They Report To and How?', 'Who Do They Interact With and In What Manner?']]

# Generate embeddings for each relevant column in the dataset
def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embedding each relevant column separately
    column_embeddings = {}
    for column in data.columns[1:]:  # Skip 'Health Body' column
        column_embeddings[column] = model.encode(data[column].tolist())

    return column_embeddings, model

# Find the best matching answer based on cosine similarity
def get_best_answer(user_query, column_embeddings, model, data, threshold=0.60):
    user_query_embedding = model.encode([user_query])

    best_similarity = -1
    best_match_entity = ""
    best_match_column = ""
    best_match_answer = ""

    # Compare user query with each column's embeddings
    for column, embeddings in column_embeddings.items():
        similarities = cosine_similarity(user_query_embedding, embeddings)
        best_match_index = similarities.argmax()
        similarity_score = similarities.max()

        # Check if this is the best match so far
        if similarity_score > best_similarity:
            best_similarity = similarity_score
            best_match_entity = data['Health Body'].iloc[best_match_index]
            best_match_column = column
            best_match_answer = data[column].iloc[best_match_index]
            print("Similarity Score", similarity_score, "Best Similarity", best_similarity)

    # Fallback if similarity is too low
    if best_similarity < threshold or not best_match_answer:
        return None, None, "Sorry, I couldn't find a relevant answer. Please try rephrasing your question or ask relevant to NHS Wales."

    return best_match_entity, best_match_column, best_match_answer


# Save embeddings for future use to avoid recalculating
def save_embeddings(column_embeddings):
    with open('column_embeddings.pkl', 'wb') as f:
        pickle.dump(column_embeddings, f)

# Load embeddings (if you want to avoid regenerating them every time)
def load_embeddings():
    with open('column_embeddings.pkl', 'rb') as f:
        return pickle.load(f)

# Main function to initialize the dataset and embeddings
def initialize():
    data = load_data()
    column_embeddings, model = generate_embeddings(data)
    save_embeddings(column_embeddings)
    return data, column_embeddings, model

if __name__ == "__main__":
    data, column_embeddings, model = initialize()
    print("Chatbot is ready!")
