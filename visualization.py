# chatgpt_cost_dashboard/visualization.py
import streamlit as st
import plotly.graph_objects as go
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class Visualization:
    def __init__(self, data_analysis):
        self.data_analysis = data_analysis

    def display_summary_statistics(self):
        """Display summary statistics."""
        conversations_df = self.data_analysis.load_conversations_data()

        col1, col2 = st.columns(2)

        # Summary Statistics
        with col1:
            # Display message count
            if "message_id" in conversations_df.columns:
                message_count = conversations_df["message_id"].nunique()
                st.metric("Total Messages", message_count)
            else:
                st.write("Message data not available.")

            # Display conversation count
            if "conversation_id" in conversations_df.columns:
                total_input_tokens = conversations_df["input_tokens"].sum()
                total_output_tokens = conversations_df["output_tokens"].sum()
                conversation_count = conversations_df["conversation_id"].nunique()
                st.metric("Total Conversations", conversation_count)
                st.metric("Total Input Tokens", f"{total_input_tokens:,}")
                st.metric("Total Output Tokens", f"{total_output_tokens:,}")
            else:
                st.write("Conversation data not available.")

        with col2:
            # Display cost summary
            if all(
                col in conversations_df.columns
                for col in [
                    "input_cost",
                    "output_cost",
                    "total_cost",
                    "input_tokens",
                    "output_tokens",
                ]
            ):
                total_input_cost = conversations_df["input_cost"].sum()
                total_output_cost = conversations_df["output_cost"].sum()
                total_cost = conversations_df["total_cost"].sum()

                st.metric("Total Input Cost", f"${total_input_cost:,.2f}")
                st.metric("Total Output Cost", f"${total_output_cost:,.2f}")
                st.metric("Total Cost", f"${total_cost:,.2f}")
            else:
                st.write("Cost data not available.")

    def display_period_breakdown(self):
        """Display period breakdown of costs."""
        conversations_df = self.data_analysis.load_conversations_data()

        if "message_create_datetime" in conversations_df.columns:
            st.write("### Period Breakdown")
            period_tabs = st.tabs(["Monthly", "Weekly", "Daily"])

            with period_tabs[0]:
                st.write("### Monthly Cost Breakdown")
                monthly_costs_df = self.data_analysis.calculate_period_costs(
                    conversations_df, "M"
                )
                st.dataframe(
                    monthly_costs_df.style.format(
                        {
                            "input_cost": "${:,.2f}",
                            "output_cost": "${:,.2f}",
                            "total_cost": "${:,.2f}",
                            "input_tokens": "{:,}",
                            "output_tokens": "{:,}",
                        }
                    ).background_gradient(subset=["total_cost"], cmap="OrRd")
                )
                st.line_chart(
                    monthly_costs_df.set_index("M")[
                        [
                            "input_tokens",
                            "output_tokens",
                            "input_cost",
                            "output_cost",
                            "total_cost",
                        ]
                    ]
                )

            with period_tabs[1]:
                st.write("### Weekly Cost Breakdown")
                weekly_costs_df = self.data_analysis.calculate_period_costs(
                    conversations_df, "W"
                )
                st.dataframe(
                    weekly_costs_df.style.format(
                        {
                            "input_cost": "${:,.2f}",
                            "output_cost": "${:,.2f}",
                            "total_cost": "${:,.2f}",
                            "input_tokens": "{:,}",
                            "output_tokens": "{:,}",
                        }
                    ).background_gradient(subset=["total_cost"], cmap="OrRd")
                )
                st.line_chart(
                    weekly_costs_df.set_index("W")[
                        [
                            "input_tokens",
                            "output_tokens",
                            "input_cost",
                            "output_cost",
                            "total_cost",
                        ]
                    ]
                )

            with period_tabs[2]:
                st.write("### Daily Cost Breakdown")
                daily_costs_df = self.data_analysis.calculate_period_costs(
                    conversations_df, "D"
                )
                st.dataframe(
                    daily_costs_df.style.format(
                        {
                            "input_cost": "${:,.2f}",
                            "output_cost": "${:,.2f}",
                            "total_cost": "${:,.2f}",
                            "input_tokens": "{:,}",
                            "output_tokens": "{:,}",
                        }
                    ).background_gradient(subset=["total_cost"], cmap="OrRd")
                )
                st.line_chart(
                    daily_costs_df.set_index("D")[
                        [
                            "input_tokens",
                            "output_tokens",
                            "input_cost",
                            "output_cost",
                            "total_cost",
                        ]
                    ]
                )

        else:
            st.write("Datetime data not available.")

    def display_sentiment_analysis(self):
        """Perform sentiment analysis and display results."""
        st.write("### Sentiment Analysis")
        messages_df = self.data_analysis.load_conversations_data()
        messages_df = self.perform_sentiment_analysis(messages_df)
        sentiment_counts = messages_df["sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

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

    def display_keyword_analysis(self):
        """Perform keyword analysis and display results."""
        st.write("### Keyword Analysis")
        messages_df = self.data_analysis.load_conversations_data()
        keyword_df = self.perform_keyword_analysis(messages_df)
        st.write(keyword_df)
        st.bar_chart(keyword_df.set_index("keyword"))

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

    def display_cost_forecasting(self):
        """Perform cost forecasting and display results."""
        st.write("### Cost Forecasting")
        messages_df = self.data_analysis.load_conversations_data()
        forecast_df = self.perform_cost_forecasting(messages_df)
        st.line_chart(forecast_df.set_index("month")["forecasted_cost"])

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

    def display_token_efficiency(self):
        """Display token efficiency."""
        st.write("### Token Efficiency")
        messages_df = self.data_analysis.load_conversations_data()
        efficiency = (
            100 * messages_df["output_tokens"].sum() / messages_df["input_tokens"].sum()
        )
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=efficiency,
                title={"text": "Token Efficiency"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 100], "color": "blue"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        st.plotly_chart(fig)
