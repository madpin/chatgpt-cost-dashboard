# chatgpt_cost_dashboard/data_analysis.py
import pandas as pd
import sqlite3
import json
import tiktoken
from datetime import datetime
import time
import os
import zipfile
import streamlit as st
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression

# Constants
DB_FILE = "./data/conversations.db"


class DataAnalysis:
    """Class responsible for data analysis and calculations."""

    def __init__(self):
        """Initialize DataAnalysis class with database file path."""
        self.db_file = DB_FILE

    def load_data(self, query):
        """Load data from SQLite database."""
        with sqlite3.connect(self.db_file) as conn:
            return pd.read_sql(query, conn)
    st.cache_data()
    def load_conversations_data(self):
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
        return self.load_data(query)

    def calculate_conversation_summary(self) -> pd.DataFrame:
        """Calculate summary statistics for each conversation."""
        df = self.load_conversations_data()
        summary_df = (
            df.groupby("conversation_id")
            .agg(
                conversation_title=("conversation_title", "first"),
                conversation_create_time=("conversation_create_time", "first"),
                num_messages=("message_id", "count"),
                total_input_tokens=("input_tokens", "sum"),
                total_output_tokens=("output_tokens", "sum"),
                total_cost=("total_cost", "sum"),
                is_archived=("is_archived", "first"),
                model_slug=("default_model_slug", "first"),
            )
            .reset_index()
        )
        summary_df["total_tokens"] = (
            summary_df["total_input_tokens"] + summary_df["total_output_tokens"]
        )
        return summary_df

    def load_conversation_messages(self, conversation_id):
        """Load messages for a specific conversation."""
        df = self.load_conversations_data()
        return df[df["conversation_id"] == conversation_id]

    def calculate_period_costs(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Calculate costs by specified period (D, W, M)."""
        df["message_create_datetime"] = pd.to_datetime(
            df["message_create_datetime"], errors="coerce"
        )
        df[period] = df["message_create_datetime"].dt.to_period(period)
        period_costs = (
            df.groupby(period)
            .agg(
                input_cost=("input_cost", "sum"),
                output_cost=("output_cost", "sum"),
                total_cost=("total_cost", "sum"),
                input_tokens=("input_tokens", "sum"),
                output_tokens=("output_tokens", "sum"),
                num_messages=("message_id", "count"),
            )
            .reset_index()
            .sort_values(by=period, ascending=False)
        )
        period_costs[period] = period_costs[period].astype(str)
        return period_costs

    def perform_sentiment_analysis(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """Perform a basic sentiment analysis based on keywords."""
        positive_keywords = ["good", "great", "excellent", "positive", "happy"]
        negative_keywords = ["bad", "terrible", "poor", "negative", "sad"]

        def get_sentiment(message):
            if message is None:
                return "neutral"
            if any(word in message for word in positive_keywords):
                return "positive"
            elif any(word in message for word in negative_keywords):
                return "negative"
            else:
                return "neutral"

        messages_df["sentiment"] = messages_df["message_content"].apply(get_sentiment)
        return messages_df

    def perform_keyword_analysis(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """Perform a basic keyword frequency analysis."""
        words = messages_df["message_content"].str.cat(sep=" ").lower().split()
        word_counts = Counter(words)
        keyword_df = (
            pd.DataFrame(word_counts.items(), columns=["keyword", "frequency"])
            .sort_values(by="frequency", ascending=False)
            .head(20)
        )
        return keyword_df

    def perform_cost_forecasting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forecast future costs using linear regression."""
        df["message_create_datetime"] = pd.to_datetime(
            df["message_create_datetime"], errors="coerce"
        )
        df["month"] = df["message_create_datetime"].dt.to_period("M").astype(str)
        monthly_costs = df.groupby("month")["total_cost"].sum().reset_index()
        monthly_costs["month"] = pd.to_datetime(monthly_costs["month"])

        # Handling missing values
        monthly_costs = monthly_costs.dropna()

        # Linear regression for forecasting
        X = np.array(
            (monthly_costs["month"] - pd.to_datetime("1970-01-01")).dt.days
        ).reshape(-1, 1)
        y = monthly_costs["total_cost"].values
        model = LinearRegression().fit(X, y)

        future_months = pd.date_range(
            monthly_costs["month"].max() + pd.offsets.MonthBegin(), periods=3, freq="MS"
        )
        future_X = np.array(
            (future_months - pd.to_datetime("1970-01-01")).days
        ).reshape(-1, 1)
        future_costs = model.predict(future_X)

        forecast_df = pd.DataFrame(
            {"month": future_months, "forecasted_cost": future_costs}
        )
        return forecast_df

    def calculate_token_efficiency(self, messages_df: pd.DataFrame) -> float:
        """Calculate token efficiency."""
        if (
            "output_tokens" in messages_df.columns
            and "input_tokens" in messages_df.columns
            and messages_df["input_tokens"].sum() != 0
        ):
            efficiency = (
                100
                * messages_df["output_tokens"].sum()
                / messages_df["input_tokens"].sum()
            )
            return efficiency
        return 0.0
