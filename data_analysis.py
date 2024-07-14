# chatgpt_cost_dashboard/data_analysis.py
import pandas as pd
import sqlite3
import json
import tiktoken
from datetime import datetime
import time
import os
import zipfile

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
