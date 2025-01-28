import streamlit as st
from groq import Groq

# Initialize the Groq client with API key
API_KEY = st.secrets["groq"]["api_key"] # Replace with your actual Groq API key
client = Groq(api_key=API_KEY)

# Function to query the Groq API
def query_groq(query: str, context: str) -> str:
    try:
        prompt = f"ప్రశ్న: {query}\nసందర్భం: {context}"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant in Telugu language."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App
st.title("Telugu AI Assistant")
st.write("Enter your query in Telugu/English and get an AI-powered response in Telugu.")

# Input Fields
query = st.text_input("Enter your Question:", "")
context = st.text_area("Enter the Context (Optional):", "")

# Submit Button
if st.button("Get Answer"):
    if not API_KEY:
        st.error("API Key is missing. Please provide a valid API key.")
    elif not query:
        st.error("Please enter a valid question.")
    else:
        st.info("Querying the Groq API...")
        answer = query_groq(query, context)
        st.success(f"Answer: {answer}")
