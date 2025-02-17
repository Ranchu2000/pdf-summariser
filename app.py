import io
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Cache PDF text extraction. We read from bytes for caching.
@st.cache_data
def extract_text_from_pdf(file_bytes):
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Cache the text chunking function.
@st.cache_data
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Initialize the summarization pipeline.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Cache summarization for each text chunk. Since the global summarizer isn't hashable,
# we provide a custom hash function for its type.
@st.cache_data(hash_funcs={type(summarizer): lambda _: None})
def get_summary(chunk):
    return summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

# Initialize session state for saved cards.
if 'saved_cards' not in st.session_state:
    st.session_state.saved_cards = []

st.title("PDF Knowledge Card Generator")

# Upload PDF file.
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file:
    # Read file bytes and extract text (cached).
    file_bytes = uploaded_file.read()
    text = extract_text_from_pdf(file_bytes)
    
    if text.strip() == "":
        st.error("No text found in the PDF. Ensure the file is not image-based.")
    else:
        # Chunk the text (cached).
        chunks = chunk_text(text, chunk_size=500)
        knowledge_cards = []
        
        st.info("Generating summaries for each section...")
        # Generate summaries using the cached summarization function.
        for idx, chunk in enumerate(chunks):
            with st.spinner(f"Summarising section {idx+1}/{len(chunks)}..."):
                summary = get_summary(chunk)
            knowledge_cards.append({
                'chunk': chunk,
                'summary': summary
            })
        
        st.success("Summarisation complete!")
        st.header("Knowledge Cards")
        
        # Display each knowledge card in an expander.
        for i, card in enumerate(knowledge_cards):
            with st.expander(f"Card {i+1}: {card['summary']}"):
                st.write("**Detailed Explanation:**")
                if st.button(f"Show full detail for card {i+1}", key=f"detail_{i}"):
                    st.write(card['chunk'])
                if st.button(f"Save this card", key=f"save_{i}"):
                    st.session_state.saved_cards.append(card)
                    st.success("Card saved!")
        
        # Display saved cards in the sidebar.
        st.sidebar.header("Saved Knowledge Cards")
        if st.session_state.saved_cards:
            for j, saved_card in enumerate(st.session_state.saved_cards):
                st.sidebar.write(f"**Card {j+1}:** {saved_card['summary']}")
        else:
            st.sidebar.write("No cards saved yet.")
