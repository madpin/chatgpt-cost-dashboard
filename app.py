import streamlit as st
import pandas as pd
import sqlite3
import json
import tiktoken
from datetime import datetime
import time
import os
import plotly.graph_objects as go
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression

# Constants
DB_FILE = "./data/conversations.db"
MODEL_NAME = "gpt-4"
INPUT_COST_PER_M = 3
OUTPUT_COST_PER_M = 15

# Initialize tiktoken encoder
encoder = tiktoken.encoding_for_model(MODEL_NAME)

# Utility functions
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(encoder.encode(text))

def safe_to_datetime(ts: float) -> str:
    """Safely convert timestamp to datetime string."""
    try:
        if ts is not None and 0 <= ts < 1e18:
            return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")
        return None
    except (OverflowError, ValueError, TypeError):
        return None

def get_content_type(content):
    """Determine content type."""
    if isinstance(content, dict) and 'content_type' in content:
        return content['content_type']
    return 'text'  # Default to 'text' if content_type is not specified

def process_conversation(conversation: dict) -> list:
    """Process a single conversation and return a list of message dictionaries."""
    conversation_id = conversation.get("id", "")
    mapping = conversation.get("mapping", {})
    conversation_messages = []

    for node in mapping.values():
        if node and node.get("message"):
            message = node["message"]
            parts = message.get("content", {}).get("parts", [])
            parts_text = "".join(
                part if isinstance(part, str) else json.dumps(part) for part in parts
            )
            content_type = get_content_type(message.get("content"))
            conversation_messages.append(
                {
                    "message_id": message.get("id", ""),
                    "create_time": message.get("create_time", 0),
                    "author_role": message["author"]["role"],
                    "tokens": count_tokens(parts_text),
                    "conversation_id": conversation_id,
                    "content_type": content_type,
                    "message_content": parts_text,
                }
            )

    return sorted(conversation_messages, key=lambda x: x["create_time"] or 0)

def calculate_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate costs based on input and output tokens."""
    df["input_cost"] = (df["input_tokens"] / 1_000_000) * INPUT_COST_PER_M
    df["output_cost"] = (df["output_tokens"] / 1_000_000) * OUTPUT_COST_PER_M
    df["total_cost"] = df["input_cost"] + df["output_cost"]
    return df

def process_json_to_sqlite(json_data, db_file):
    """Process JSON data and store it in an SQLite database."""
    start_time = time.time()

    # Process data into DataFrame
    rows = [msg for conv in json_data for msg in process_conversation(conv)]
    df = pd.DataFrame(rows)
    df["create_datetime"] = pd.to_datetime(df["create_time"].apply(safe_to_datetime))

    # Calculate cumulative tokens
    df["cumulative_tokens"] = df.groupby("conversation_id")["tokens"].cumsum()
    df["input_tokens"] = df["cumulative_tokens"]
    df["output_tokens"] = df.apply(
        lambda x: x["tokens"] if x["author_role"] == "assistant" else 0, axis=1
    )

    # Calculate costs
    df = calculate_costs(df)

    # Save DataFrame to SQLite database
    connection = sqlite3.connect(db_file)
    df = df.astype(
        {
            "message_id": str,
            "conversation_id": str,
            "author_role": str,
            "create_datetime": str,
            "input_cost": float,
            "output_cost": float,
            "total_cost": float,
            "content_type": str,
            "message_content": str,
        }
    )
    df.to_sql("messages", connection, if_exists="replace", index=False)

    # Create conversations table and insert data
    conversations = [
        (
            conv["id"],
            conv["title"],
            safe_to_datetime(conv["create_time"]),
            safe_to_datetime(conv["update_time"]),
            conv["current_node"],
            int(conv["is_archived"]),
            conv.get("default_model_slug"),
        )
        for conv in json_data
    ]
    connection.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        create_time TEXT,
        update_time TEXT,
        current_node TEXT,
        is_archived INTEGER,
        default_model_slug TEXT
    )
    """)
    connection.executemany(
        """
    INSERT OR REPLACE INTO conversations (id, title, create_time, update_time, current_node, is_archived, default_model_slug)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        conversations,
    )

    connection.commit()
    connection.close()
    print(f"Data successfully imported to {db_file} in {time.time() - start_time:.2f} seconds")

def load_data(query):
    """Load data from SQLite database."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql(query, conn)

def load_conversations_data():
    """Load data for Conversations and Messages."""
    query = """
    SELECT 
        c.id as conversation_id, c.title as conversation_title, c.create_time as conversation_create_time, 
        c.update_time as conversation_update_time, c.current_node, c.is_archived, c.default_model_slug,
        m.message_id, m.create_time as message_create_time, m.author_role, m.tokens, m.conversation_id as message_conversation_id,
        m.content_type, m.message_content, m.create_datetime as message_create_datetime, m.cumulative_tokens, 
        m.input_tokens, m.output_tokens, m.input_cost, m.output_cost, m.total_cost
    FROM conversations c
    LEFT JOIN messages m ON c.id = m.conversation_id
    """
    return load_data(query)

def calculate_conversation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for each conversation."""
    summary_df = df.groupby('conversation_id').agg(
        conversation_title=('conversation_title', 'first'),
        conversation_create_time=('conversation_create_time', 'first'),
        num_messages=('message_id', 'count'),
        total_input_tokens=('input_tokens', 'sum'),
        total_output_tokens=('output_tokens', 'sum'),
        total_cost=('total_cost', 'sum')
    ).reset_index()
    summary_df['total_tokens'] = summary_df['total_input_tokens'] + summary_df['total_output_tokens']
    return summary_df

def calculate_period_costs(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Calculate costs by specified period (D, W, M)."""
    df["message_create_datetime"] = pd.to_datetime(df["message_create_datetime"], errors="coerce")
    df[period] = df["message_create_datetime"].dt.to_period(period)
    period_costs = (
        df.groupby(period)
        .agg(
            input_cost=("input_cost", "sum"),
            output_cost=("output_cost", "sum"),
            total_cost=("total_cost", "sum"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            num_messages=("message_id", "count")
        )
        .reset_index()
        .sort_values(by=period, ascending=False)
    )
    period_costs[period] = period_costs[period].astype(str)
    return period_costs

def sentiment_analysis(messages_df: pd.DataFrame) -> pd.DataFrame:
    """Perform a basic sentiment analysis based on keywords."""
    positive_keywords = ['good', 'great', 'excellent', 'positive', 'happy']
    negative_keywords = ['bad', 'terrible', 'poor', 'negative', 'sad']

    def get_sentiment(message):
        if message is None:
            return 'neutral'
        if any(word in message for word in positive_keywords):
            return 'positive'
        elif any(word in message for word in negative_keywords):
            return 'negative'
        else:
            return 'neutral'

    messages_df['sentiment'] = messages_df['message_content'].apply(get_sentiment)
    return messages_df

def keyword_analysis(messages_df: pd.DataFrame) -> pd.DataFrame:
    """Perform a basic keyword frequency analysis."""
    words = messages_df['message_content'].str.cat(sep=' ').lower().split()
    word_counts = Counter(words)
    keyword_df = pd.DataFrame(word_counts.items(), columns=['keyword', 'frequency']).sort_values(by='frequency', ascending=False).head(20)
    return keyword_df

def cost_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """Forecast future costs using linear regression."""
    df['message_create_datetime'] = pd.to_datetime(df['message_create_datetime'], errors='coerce')
    df['month'] = df['message_create_datetime'].dt.to_period('M').astype(str)
    monthly_costs = df.groupby('month')['total_cost'].sum().reset_index()
    monthly_costs['month'] = pd.to_datetime(monthly_costs['month'])

    # Handling missing values
    monthly_costs = monthly_costs.dropna()

    # Linear regression for forecasting
    X = np.array((monthly_costs['month'] - pd.to_datetime('1970-01-01')).dt.days).reshape(-1, 1)
    y = monthly_costs['total_cost'].values
    model = LinearRegression().fit(X, y)

    future_months = pd.date_range(monthly_costs['month'].max() + pd.offsets.MonthBegin(), periods=3, freq='MS')
    future_X = np.array((future_months - pd.to_datetime('1970-01-01')).days).reshape(-1, 1)
    future_costs = model.predict(future_X)

    forecast_df = pd.DataFrame({'month': future_months, 'forecasted_cost': future_costs})
    return forecast_df

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
    
    uploaded_file = st.sidebar.file_uploader("Choose your conversations.json file", type="json")
    if uploaded_file is not None:
        json_data = json.load(uploaded_file)
        
        # Ensure the data directory exists
        os.makedirs("./data", exist_ok=True)
        
        # Process the uploaded JSON file
        process_json_to_sqlite(json_data, DB_FILE)
        st.sidebar.success("File successfully processed. You can now view the dashboard.")
        st.session_state.file_processed = True
        
        # Display general information
        num_conversations = len(json_data)
        num_messages = sum(len(conv['mapping']) for conv in json_data)
        st.sidebar.write(f"**Total Conversations:** {num_conversations}")
        st.sidebar.write(f"**Total Messages:** {num_messages}")
        
        return True
    return False

def main_dashboard():
    """Render the main dashboard."""
    st.title("ChatGPT Cost Dashboard")

    # Define tabs
    tabs = st.tabs(["Conversations", "Statistics", "Advanced Analytics"])

    # Conversations Tab
    with tabs[0]:
        st.header("Conversations and Messages")
        conversations_df = load_conversations_data()

        if not conversations_df.empty and 'conversation_id' in conversations_df.columns:
            # Calculate conversation summary
            conversation_summary_df = calculate_conversation_summary(conversations_df)
            
            # Display conversations as a table
            st.write("### Conversations Data")
            st.dataframe(conversation_summary_df.style.format({
                "total_input_tokens": "{:,}",
                "total_output_tokens": "{:,}",
                "total_cost": "${:,.2f}",
                "total_tokens": "{:,}"
            }))

            # Select a conversation
            selected_conversation = st.selectbox("Select a Conversation", conversation_summary_df['conversation_id'])

            if selected_conversation:
                messages_df = conversations_df[conversations_df['conversation_id'] == selected_conversation]
                st.write("### Messages Data")
                st.dataframe(messages_df)
        else:
            st.write("No conversations data found or incorrect columns loaded.")

    # Statistics Tab
    with tabs[1]:
        st.header("Statistics")
        # st.write("### Summary Statistics")

        col1, col2 = st.columns(2)

        # Summary Statistics
        with col1:
            # Display message count
            if 'message_id' in conversations_df.columns:
                message_count = conversations_df['message_id'].nunique()
                st.metric("Total Messages", message_count)
            else:
                st.write("Message data not available.")

            # Display conversation count
            if 'conversation_id' in conversations_df.columns:
                total_input_tokens = conversations_df["input_tokens"].sum()
                total_output_tokens = conversations_df["output_tokens"].sum()
                conversation_count = conversations_df['conversation_id'].nunique()
                st.metric("Total Conversations", conversation_count)
                st.metric("Total Input Tokens", f"{total_input_tokens:,}")
                st.metric("Total Output Tokens", f"{total_output_tokens:,}")
            else:
                st.write("Conversation data not available.")

        with col2:
            # Display cost summary
            if all(col in conversations_df.columns for col in ["input_cost", "output_cost", "total_cost", "input_tokens", "output_tokens"]):
                total_input_cost = conversations_df["input_cost"].sum()
                total_output_cost = conversations_df["output_cost"].sum()
                total_cost = conversations_df["total_cost"].sum()

                st.metric("Total Input Cost", f"${total_input_cost:,.2f}")
                st.metric("Total Output Cost", f"${total_output_cost:,.2f}")
                st.metric("Total Cost", f"${total_cost:,.2f}")
            else:
                st.write("Cost data not available.")

        # Period Breakdown Tabs
        if 'message_create_datetime' in conversations_df.columns:
            st.write("### Period Breakdown")
            period_tabs = st.tabs(["Daily", "Weekly", "Monthly"])

            with period_tabs[0]:
                st.write("### Daily Cost Breakdown")
                daily_costs_df = calculate_period_costs(conversations_df, "D")
                st.dataframe(daily_costs_df.style.format({
                    "input_cost": "${:,.2f}",
                    "output_cost": "${:,.2f}",
                    "total_cost": "${:,.2f}",
                    "input_tokens": "{:,}",
                    "output_tokens": "{:,}"
                }).background_gradient(subset=["total_cost"], cmap="OrRd"))
                st.line_chart(daily_costs_df.set_index("D")[["input_tokens", "output_tokens", "input_cost", "output_cost", "total_cost"]])

            with period_tabs[1]:
                st.write("### Weekly Cost Breakdown")
                weekly_costs_df = calculate_period_costs(conversations_df, "W")
                st.dataframe(weekly_costs_df.style.format({
                    "input_cost": "${:,.2f}",
                    "output_cost": "${:,.2f}",
                    "total_cost": "${:,.2f}",
                    "input_tokens": "{:,}",
                    "output_tokens": "{:,}"
                }).background_gradient(subset=["total_cost"], cmap="OrRd"))
                st.line_chart(weekly_costs_df.set_index("W")[["input_tokens", "output_tokens", "input_cost", "output_cost", "total_cost"]])

            with period_tabs[2]:
                st.write("### Monthly Cost Breakdown")
                monthly_costs_df = calculate_period_costs(conversations_df, "M")
                st.dataframe(monthly_costs_df.style.format({
                    "input_cost": "${:,.2f}",
                    "output_cost": "${:,.2f}",
                    "total_cost": "${:,.2f}",
                    "input_tokens": "{:,}",
                    "output_tokens": "{:,}"
                }).background_gradient(subset=["total_cost"], cmap="OrRd"))
                st.line_chart(monthly_costs_df.set_index("M")[["input_tokens", "output_tokens", "input_cost", "output_cost", "total_cost"]])
        else:
            st.write("Datetime data not available.")

    # Advanced Analytics Tab
    with tabs[2]:
        st.header("Advanced Analytics")

        col1, col2 = st.columns(2)

        # Sentiment Analysis
        with col1:
            st.write("### Sentiment Analysis")
            messages_df = load_conversations_data()
            messages_df = sentiment_analysis(messages_df)
            sentiment_counts = messages_df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

        # Keyword Analysis
        with col2:
            st.write("### Keyword Analysis")
            keyword_df = keyword_analysis(messages_df)
            st.write(keyword_df)
            st.bar_chart(keyword_df.set_index('keyword'))

        # Cost Forecasting
        st.write("### Cost Forecasting")
        forecast_df = cost_forecasting(messages_df)
        st.line_chart(forecast_df.set_index('month')['forecasted_cost'])

        # Token Efficiency
        st.write("### Token Efficiency")
        efficiency = 100 * messages_df['output_tokens'].sum() / messages_df['input_tokens'].sum()
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=efficiency,
            title={'text': "Token Efficiency"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 100], 'color': "blue"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))

        st.plotly_chart(fig)

def main():
    """Main application logic."""
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False

    file_uploaded = sidebar()

    if st.session_state.file_processed:
        main_dashboard()


if __name__ == "__main__":
    main()
