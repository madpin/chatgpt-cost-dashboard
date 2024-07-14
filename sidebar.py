# chatgpt_cost_dashboard/sidebar.py
import streamlit as st
import json
import zipfile
import os
from data_ingestion import DataIngestion

def sidebar():
    """Render the sidebar."""
    st.sidebar.title("ChatGPT Cost")

    # Instructions expander
    with st.sidebar.expander("Instructions"):
        st.markdown("""
        ### Instructions to get the `conversations.json` file:

        1. Visit the ChatGPT website.
        2. Request an export of your conversation data.
        3. You will receive a link in your email to download the exported data.
        4. Download the email, unzip the file, and locate the `conversations.json` file.
        5. Upload the `conversations.json` file into this tool.
        """)
        
    st.sidebar.subheader("Upload Conversations File")
    
    uploaded_file = st.sidebar.file_uploader("Choose your conversations.zip or conversations.json file", type=["zip", "json"])
    if uploaded_file is not None:
        # Ensure the data directory exists
        os.makedirs("./data", exist_ok=True)

        # Initialize data processor
        data_ingestion = DataIngestion()

        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall("./data")
                # Look for conversations.json in the extracted files
                extracted_files = zip_ref.namelist()
                if "conversations.json" in extracted_files:
                    with open("./data/conversations.json", "r") as json_file:
                        json_data = json.load(json_file)
                        data_ingestion.process_json_to_sqlite(json_data)
                        st.sidebar.success("File successfully processed. You can now view the dashboard.")
                        st.session_state.file_processed = True

        elif uploaded_file.name.endswith('.json'):
            json_data = json.load(uploaded_file)
            data_ingestion.process_json_to_sqlite(json_data)
            st.sidebar.success("File successfully processed. You can now view the dashboard.")
            st.session_state.file_processed = True
        
        # Display general information
        num_conversations = len(json_data)
        num_messages = sum(len(conv['mapping']) for conv in json_data)
        st.sidebar.write(f"**Total Conversations:** {num_conversations}")
        st.sidebar.write(f"**Total Messages:** {num_messages}")
        
        return True
    return False
