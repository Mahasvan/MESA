import streamlit as st
import requests

# API endpoint configuration
API_HOST = st.sidebar.text_input("API Host", "localhost")
API_PORT = st.sidebar.text_input("API Port", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/bert/predict"

# App title and description
st.title("Toxicity Detector")
st.write("Enter a sentence to check if it's toxic or not.")

# Input text box
user_input = st.text_area("Enter your text here:", height=150)

# Submit button
if st.button("Check Toxicity"):
    if user_input:
        try:
            # Call the API
            response = requests.get(API_URL, params={"message": user_input})
            
            if response.status_code == 200:
                result = response.json()["response"]
                # Display result
                if result == 0:
                    st.success("✅ Not Toxic")
                    st.write("The message is not considered toxic.")
                elif result == 1:
                    st.error("⚠️ Toxic")
                    st.write("The message is considered toxic.")
                else:
                    st.warning("Unexpected response from API")
            else:
                st.error(f"Error: API returned status code {response.status_code}")
                st.write(f"Response: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
            st.info("Make sure the API is running at the specified host and port.")
    else:
        st.warning("Please enter some text to analyze.")

# Instructions
with st.expander("How to use"):
    st.write("""
    1. Enter a sentence in the text box above.
    2. Click the 'Check Toxicity' button.
    3. The app will display whether the sentence is toxic or not.
    
    You can change the API host and port in the sidebar if needed.
    """)