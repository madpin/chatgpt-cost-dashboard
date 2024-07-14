import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from data_analysis import DataAnalysis
from data_ingestion import DataIngestion
from visualization import Visualization
from sidebar import sidebar


# Set the page layout to wide
st.set_page_config(layout="wide")


def main_dashboard():
    """Render the main dashboard."""
    st.title("ChatGPT Cost Dashboard")

    # Define tabs
    tabs = st.tabs(["Conversations", "Statistics", "Advanced Analytics"])

    # Initialize data processor
    data_analysis = DataAnalysis()
    data_ingestion = DataIngestion()

    # Conversations Tab
    with tabs[0]:
        st.header("Conversations and Messages")

        # Load and display conversation summary
        conversation_summary_df = data_analysis.calculate_conversation_summary()
        if not conversation_summary_df.empty:
            st.write("### Conversations Data")

            # Setup AgGrid options for single row selection
            gb = GridOptionsBuilder.from_dataframe(conversation_summary_df)
            gb.configure_selection("single")
            grid_options = gb.build()

            # Display the dataframe using AgGrid
            grid_response = AgGrid(
                conversation_summary_df,
                gridOptions=grid_options,
                height=300,
                fit_columns_on_grid_load=True,
            )
            selected_rows = None
            # Check if any rows are selected
            selected_rows: pd.DataFrame = grid_response.get("selected_rows", None)
            # print(selected_rows)
            if selected_rows is not None:
                selected_conversation = selected_rows["conversation_id"][0]

                messages_df = data_analysis.load_conversation_messages(
                    selected_conversation
                )
                st.write(f"### Messages Data for Conversation: {selected_conversation}")
                st.dataframe(
                    messages_df.drop(
                        columns=[
                            "conversation_id",
                            "conversation_title",
                            "conversation_create_time",
                            "conversation_update_time",
                            "current_node",
                            "is_archived",
                            "default_model_slug",
                        ]
                    ).style.format(
                        {
                            "input_cost": "${:,.5f}",
                            "output_cost": "${:,.5f}",
                            "total_cost": "${:,.5f}",
                        }),
                    use_container_width=True,
                )
            else:
                st.write(
                    "Select a conversation from the table above to view its messages."
                )
        else:
            st.write("No conversations data found or incorrect columns loaded.")

    # Statistics Tab
    with tabs[1]:
        st.header("Statistics")

        # Initialize Visualization object
        visualization = Visualization(data_analysis)

        # Display summary statistics
        visualization.display_summary_statistics()

        # Period Breakdown
        visualization.display_period_breakdown()

    # Advanced Analytics Tab
    with tabs[2]:
        st.header("Advanced Analytics")

        # Sentiment Analysis
        visualization.display_sentiment_analysis()

        # Keyword Analysis
        visualization.display_keyword_analysis()

        # Cost Forecasting
        visualization.display_cost_forecasting()

        # Token Efficiency
        visualization.display_token_efficiency()


def main():
    """Main application logic."""
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False

    file_uploaded = sidebar()

    if st.session_state.file_processed:
        main_dashboard()


if __name__ == "__main__":
    main()
