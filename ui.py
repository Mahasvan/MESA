import requests
import streamlit as st

# API endpoint configuration
API_HOST = st.sidebar.text_input("API Host", "localhost")
API_PORT = st.sidebar.text_input("API Port", "8000")

# Model selection
st.sidebar.markdown("---")
st.sidebar.subheader("Model Selection")
model_choice = st.sidebar.radio(
    "Choose model(s) to use:",
    ("BERT", "CNN", "Both"),
    index=0
)

# App title and description
st.title("Toxicity Detector")
st.write("Enter a sentence to check if it's toxic or not.")

# Input text box
user_input = st.text_area("Enter your text here:", height=150)


# Helper function to call API endpoint
def call_api(endpoint, message):
    """Call a specific API endpoint and return the result"""
    api_url = f"http://{API_HOST}:{API_PORT}/{endpoint}/predict"
    response = requests.get(api_url, params={"message": message})

    if response.status_code == 200:
        return response.json()["response"], None
    else:
        return None, f"API returned status code {response.status_code}: {response.text}"


# Helper function to display result
def display_result(result, model_name=""):
    """Display the toxicity result"""
    prefix = f"**{model_name}**: " if model_name else ""
    if result == 0:
        st.success(f"✅ {prefix}Not Toxic")
        st.write(f"{prefix}The message is not considered toxic.")
    elif result == 1:
        st.error(f"⚠️ {prefix}Toxic")
        st.write(f"{prefix}The message is considered toxic.")
    else:
        st.warning(f"{prefix}Unexpected response from API")


# Submit button
if st.button("Check Toxicity"):
    if user_input:
        try:
            if model_choice == "BERT":
                result, error = call_api("bert", user_input)
                if error:
                    st.error(f"Error: {error}")
                else:
                    display_result(result)

            elif model_choice == "CNN":
                result, error = call_api("cnn", user_input)
                if error:
                    st.error(f"Error: {error}")
                else:
                    display_result(result)

            elif model_choice == "Both":
                st.subheader("Results from both models:")

                # Call BERT endpoint
                bert_result, bert_error = call_api("bert", user_input)
                if bert_error:
                    st.error(f"BERT Error: {bert_error}")
                else:
                    display_result(bert_result, "BERT")

                st.markdown("---")

                # Call CNN endpoint
                cnn_result, cnn_error = call_api("cnn", user_input)
                if cnn_error:
                    st.error(f"CNN Error: {cnn_error}")
                else:
                    display_result(cnn_result, "CNN")

                # Show comparison if both succeeded
                if not bert_error and not cnn_error:
                    st.markdown("---")
                    if bert_result == cnn_result:
                        st.info("🤝 Both models agree on the result!")
                    else:
                        st.warning("⚖️ Models disagree - consider the context and use your judgment.")

        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
            st.info("Make sure the API is running at the specified host and port.")
    else:
        st.warning("Please enter some text to analyze.")

# Instructions
with st.expander("How to use"):
    st.write("""
    1. Select your preferred model(s) from the sidebar:
       - **BERT**: Use the BERT-based toxicity detection model
       - **CNN**: Use the CNN-based toxicity detection model  
       - **Both**: Compare results from both models
    2. Enter a sentence in the text box above.
    3. Click the 'Check Toxicity' button.
    4. The app will display whether the sentence is toxic or not.

    When using "Both" models, you'll see results from each model and whether they agree.
    You can change the API host and port in the sidebar if needed.
    """)
