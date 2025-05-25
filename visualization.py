# chatgpt_cost_dashboard/visualization.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


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
        # Ensure 'message_content' column exists and is not empty
        if "message_content" in messages_df.columns and not messages_df["message_content"].isnull().all():
            messages_df = self.data_analysis.perform_sentiment_analysis(messages_df)
            sentiment_counts = messages_df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)
        else:
            st.write("Sentiment analysis cannot be performed due to missing or empty message content.")

    def display_keyword_analysis(self):
        """Perform keyword analysis and display results."""
        st.write("### Keyword Analysis")
        messages_df = self.data_analysis.load_conversations_data()
        # Ensure 'message_content' column exists and is not empty
        if "message_content" in messages_df.columns and not messages_df["message_content"].isnull().all():
            keyword_df = self.data_analysis.perform_keyword_analysis(messages_df)
            st.write(keyword_df)
            st.bar_chart(keyword_df.set_index("keyword"))
        else:
            st.write("Keyword analysis cannot be performed due to missing or empty message content.")

    def display_cost_forecasting(self):
        """Perform cost forecasting and display results."""
        st.write("### Cost Forecasting")
        messages_df = self.data_analysis.load_conversations_data()
        # Ensure necessary columns exist for forecasting
        if "message_create_datetime" in messages_df.columns and "total_cost" in messages_df.columns:
            forecast_df = self.data_analysis.perform_cost_forecasting(messages_df)
            if not forecast_df.empty:
                st.line_chart(forecast_df.set_index("month")["forecasted_cost"])
            else:
                st.write("Cost forecasting could not be performed. The data might be insufficient.")
        else:
            st.write("Cost forecasting cannot be performed due to missing datetime or cost data.")

    def display_token_efficiency(self):
        """Display token efficiency."""
        st.write("### Token Efficiency")
        messages_df = self.data_analysis.load_conversations_data()
        efficiency = self.data_analysis.calculate_token_efficiency(messages_df)
        
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
                        "value": 90,  # Target efficiency
                    },
                },
            )
        )
        st.plotly_chart(fig)
