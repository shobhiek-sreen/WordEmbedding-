
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertModel
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model and tokenizer for sentence embeddings
tokenizer_emb = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model_emb = DistilBertModel.from_pretrained('distilbert-base-cased')


def encode_sentence_sim(sentence, max_length=512):
    # Tokenize input
    inputs = tokenizer_emb(sentence, return_tensors='pt', truncation=True, max_length=max_length)

    # Get model output
    outputs = model_emb(**inputs)

    # Extract the embeddings from 'last_hidden_state'
    embeddings = outputs.last_hidden_state

    # Reduce embeddings to a fixed size (e.g., by mean pooling)
    embeddings = torch.mean(embeddings, dim=1)

    return embeddings

def visualize_word_embeddings(embeddings, tokens):
    # Ensure embeddings is a 2D array
    embeddings = embeddings.reshape((len(tokens), -1))

    # Use t-SNE to reduce dimensionality for visualization
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(5, len(tokens)-1))
    word_vectors_2D = tsne_model.fit_transform(embeddings)

    # Plotting the words in 2D
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, token in enumerate(tokens):
        ax.scatter(word_vectors_2D[i, 0], word_vectors_2D[i, 1], marker='o', color='b')
        ax.text(word_vectors_2D[i, 0] + 0.02, word_vectors_2D[i, 1] + 0.02, token, fontsize=9)

    # Pass the figure to st.pyplot()
    st.pyplot(fig)


def encode_sentence(sentence):
    # Tokenize input
    inputs = tokenizer_emb(sentence, return_tensors='pt')

    # Get model output
    outputs = model_emb(**inputs)

    # Extract the embeddings from 'last_hidden_state'
    embeddings = outputs.last_hidden_state

    return embeddings

def calculate_cosine_similarity(embeddings1, embeddings2):
    # Detach tensors before converting to numpy arrays
    embeddings1 = embeddings1.detach().numpy()
    embeddings2 = embeddings2.detach().numpy()

    # Reshape embeddings if needed
    embeddings1 = embeddings1.reshape(1, -1)
    embeddings2 = embeddings2.reshape(1, -1)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    # Extract the similarity score
    similarity_score = similarity_matrix[0, 0]

    return similarity_score

# Streamlit app
def main():
    st.title(" Word Embeddings Demo and  Similarity between Sentence")

    # Input for QA
    firstsentence = st.text_area("Enter First Sentence:")
    secondsentence = st.text_area("Enter Second Sentence:")


    # Use the entered question for sentence embeddings visualization
    st.header("Sentence Embeddings")
    if st.button("Visualize Embeddings"):
        embeddings = encode_sentence(secondsentence)  # Use the entered question for embeddings
        tokens = tokenizer_emb.tokenize(tokenizer_emb.decode(tokenizer_emb.encode(secondsentence)))
        st.header("Visualization of Word Embeddings")
        visualize_word_embeddings(embeddings[0].detach().numpy(), tokens)

    # Calculate cosine similarity
    st.header("Similarity Between Sentence")
    context_embeddings = encode_sentence_sim(firstsentence)
    question_embeddings = encode_sentence_sim(secondsentence)
    similarity_score = calculate_cosine_similarity(context_embeddings, question_embeddings)
    st.write(f"Cosine Similarity between context and question: {similarity_score}")

if __name__ == "__main__":
    main()
