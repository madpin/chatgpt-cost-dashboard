# # chatgpt_cost_dashboard/data_processing.py
# import pandas as pd
# import sqlite3
# import json
# import tiktoken
# from datetime import datetime
# import time
# import os
# import zipfile

# # Constants
# DB_FILE = "./data/conversations.db"
# MODEL_NAME = "gpt-4"
# INPUT_COST_PER_M = 3
# OUTPUT_COST_PER_M = 15

# # Initialize tiktoken encoder
# encoder = tiktoken.encoding_for_model(MODEL_NAME)

# class DataProcessor:
#     def __init__(self):
#         self.db_file = DB_FILE

#     def count_tokens(self, text: str) -> int:
#         """Count tokens using tiktoken."""
#         return len(encoder.encode(text))

#     def safe_to_datetime(self, ts: float) -> str:
#         """Safely convert timestamp to datetime string."""
#         try:
#             if ts is not None and 0 <= ts < 1e18:
#                 return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")
#             return None
#         except (OverflowError, ValueError, TypeError):
#             return None

#     def get_content_type(self, content):
#         """Determine content type."""
#         if isinstance(content, dict) and 'content_type' in content:
#             return content['content_type']
#         return 'text'  # Default to 'text' if content_type is not specified

#     def process_conversation(self, conversation: dict) -> list:
#         """Process a single conversation and return a list of message dictionaries."""
#         conversation_id = conversation.get("id", "")
#         mapping = conversation.get("mapping", {})
#         conversation_messages = []

#         for node in mapping.values():
#             if node and node.get("message"):
#                 message = node["message"]
#                 parts = message.get("content", {}).get("parts", [])
#                 parts_text = "".join(
#                     part if isinstance(part, str) else json.dumps(part) for part in parts
#                 )
#                 content_type = self.get_content_type(message.get("content"))
#                 conversation_messages.append(
#                     {
#                         "message_id": message.get("id", ""),
#                         "create_time": message.get("create_time", 0),
#                         "author_role": message["author"]["role"],
#                         "tokens": self.count_tokens(parts_text),
#                         "conversation_id": conversation_id,
#                         "content_type": content_type,
#                         "message_content": parts_text,
#                     }
#                 )

#         return sorted(conversation_messages, key=lambda x: x["create_time"] or 0)

#     def calculate_costs(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Calculate costs based on input and output tokens."""
#         df["input_cost"] = (df["input_tokens"] / 1_000_000) * INPUT_COST_PER_M
#         df["output_cost"] = (df["output_tokens"] / 1_000_000) * OUTPUT_COST_PER_M
#         df["total_cost"] = df["input_cost"] + df["output_cost"]
#         return df

#     def process_json_to_sqlite(self, json_data):
#         """Process JSON data and store it in an SQLite database."""
#         start_time = time.time()

#         # Process data into DataFrame
#         rows = [msg for conv in json_data for msg in self.process_conversation(conv)]
#         df = pd.DataFrame(rows)
#         df["create_datetime"] = pd.to_datetime(df["create_time"].apply(self.safe_to_datetime))

#         # Calculate cumulative tokens
#         df["cumulative_tokens"] = df.groupby("conversation_id")["tokens"].cumsum()
#         df["input_tokens"] = df["cumulative_tokens"]
#         df["output_tokens"] = df.apply(
#             lambda x: x["tokens"] if x["author_role"] == "assistant" else 0, axis=1
#         )

#         # Calculate costs
#         df = self.calculate_costs(df)

#         # Save DataFrame to SQLite database
#         connection = sqlite3.connect(self.db_file)
#         df = df.astype(
#             {
#                 "message_id": str,
#                 "conversation_id": str,
#                 "author_role": str,
#                 "create_datetime": str,
#                 "input_cost": float,
#                 "output_cost": float,
#                 "total_cost": float,
#                 "content_type": str,
#                 "message_content": str,
#             }
#         )
#         df.to_sql("messages", connection, if_exists="replace", index=False)

#         # Create conversations table and insert data
#         conversations = [
#             (
#                 conv["id"],
#                 conv["title"],
#                 self.safe_to_datetime(conv["create_time"]),
#                 self.safe_to_datetime(conv["update_time"]),
#                 conv["current_node"],
#                 int(conv["is_archived"]),
#                 conv.get("default_model_slug"),
#             )
#             for conv in json_data
#         ]
#         connection.execute("""
#         CREATE TABLE IF NOT EXISTS conversations (
#             id TEXT PRIMARY KEY,
#             title TEXT,
#             create_time TEXT,
#             update_time TEXT,
#             current_node TEXT,
#             is_archived INTEGER,
#             default_model_slug TEXT
#         )
#         """)
#         connection.executemany(
#             """
#         INSERT OR REPLACE INTO conversations (id, title, create_time, update_time, current_node, is_archived, default_model_slug)
#         VALUES (?, ?, ?, ?, ?, ?, ?)
#         """,
#             conversations,
#         )

#         connection.commit()
#         connection.close()
#         print(f"Data successfully imported to {self.db_file} in {time.time() - start_time:.2f} seconds")

#     def load_data(self, query):
#         """Load data from SQLite database."""
#         with sqlite3.connect(self.db_file) as conn:
#             return pd.read_sql(query, conn)

#     def load_conversations_data(self):
#         """Load data for Conversations and Messages."""
#         query = """
#         SELECT 
#             c.id as conversation_id, c.title as conversation_title, c.create_time as conversation_create_time, 
#             c.update_time as conversation_update_time, c.current_node, c.is_archived, c.default_model_slug,
#             m.message_id, m.create_time as message_create_time, m.author_role, m.tokens, m.conversation_id as message_conversation_id,
#             m.content_type, m.message_content, m.create_datetime as message_create_datetime, m.cumulative_tokens, 
#             m.input_tokens, m.output_tokens, m.input_cost, m.output_cost, m.total_cost
#         FROM conversations c
#         LEFT JOIN messages m ON c.id = m.conversation_id
#         """
#         return self.load_data(query)

#     def calculate_conversation_summary(self) -> pd.DataFrame:
#         """Calculate summary statistics for each conversation."""
#         df = self.load_conversations_data()
#         summary_df = df.groupby('conversation_id').agg(
#             conversation_title=('conversation_title', 'first'),
#             conversation_create_time=('conversation_create_time', 'first'),
#             num_messages=('message_id', 'count'),
#             total_input_tokens=('input_tokens', 'sum'),
#             total_output_tokens=('output_tokens', 'sum'),
#             total_cost=('total_cost', 'sum')
#         ).reset_index()
#         summary_df['total_tokens'] = summary_df['total_input_tokens'] + summary_df['total_output_tokens']
#         return summary_df

#     def load_conversation_messages(self, conversation_id):
#         """Load messages for a specific conversation."""
#         df = self.load_conversations_data()
#         return df[df['conversation_id'] == conversation_id]

#     def calculate_period_costs(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
#         """Calculate costs by specified period (D, W, M)."""
#         df["message_create_datetime"] = pd.to_datetime(df["message_create_datetime"], errors="coerce")
#         df[period] = df["message_create_datetime"].dt.to_period(period)
#         period_costs = (
#             df.groupby(period)
#             .agg(
#                 input_cost=("input_cost", "sum"),
#                 output_cost=("output_cost", "sum"),
#                 total_cost=("total_cost", "sum"),
#                 input_tokens=("input_tokens", "sum"),
#                 output_tokens=("output_tokens", "sum"),
#                 num_messages=("message_id", "count")
#             )
#             .reset_index()
#             .sort_values(by=period, ascending=False)
#         )
#         period_costs[period] = period_costs[period].astype(str)
#         return period_costs
