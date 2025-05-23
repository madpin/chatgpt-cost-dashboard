# ChatGPT Cost Dashboard

## Overview

The **ChatGPT Cost Dashboard** is a comprehensive tool designed to process, analyze, and visualize the costs associated with using OpenAI's ChatGPT if that were to be done by the API.  
This Streamlit application allows users to upload JSON files containing conversation data, which are then processed and stored in an SQLite database.  
The dashboard provides detailed insights into token usage, costs, and various analytical metrics, making it easier to manage and optimize the usage of ChatGPT.

You can find the application [here](https://chatgpt-cost-dashboard.streamlit.app/).
## Features

- **Data Upload and Processing**: Upload JSON files containing conversation data, which are processed and stored in an SQLite database.
- **Conversations Overview**: View detailed information about each conversation, including the number of messages, token usage, and costs.
- **Statistics and Metrics**: Get key metrics such as total messages, total conversations, input tokens, output tokens, and associated costs.
- **Period Breakdown**: Analyze costs over different periods (daily, weekly, monthly) with interactive charts.
- **Advanced Analytics**: Perform sentiment analysis, keyword frequency analysis, and cost forecasting.
- **Interactive Visualizations**: Use Plotly for advanced visualizations and Streamlit for an interactive user interface.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/madpin/chatgpt-cost-dashboard.git
   cd chatgpt-cost-dashboard
   ```

2. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```sh
   streamlit run dashboard.py
   ```

## Usage

1. **Upload Conversations File**: Use the sidebar to upload your `conversations.json` file.
<!-- ![Input where you should add the file](static/img/input_page.png) -->
   <img src="static/img/input_page.png" alt="Input where you should add the file" style="max-width: 400px;" />  

2. **View Dashboard**: Once the file is processed, navigate through the tabs to explore different aspects of the conversation data.
   - **Conversations Tab**: View detailed conversation data and select specific conversations for more insights.
   <!-- ![Messages](static/img/messages.png) -->
   <img src="static/img/messages.png" alt="Messages" style="max-width: 400px;" />  
   
   - **Statistics Tab**: See summary statistics and breakdowns of costs over different periods.
   <!-- ![Statistics](static/img/statistics_monthly.png) -->
   <img src="static/img/statistics_monthly.png" alt="Statistics" style="max-width: 400px;" />

   - **Advanced Analytics Tab**: Explore advanced analytics including sentiment and keyword analysis, and cost forecasting.

## File Structure

- `dashboard.py`: The main Streamlit application file. Handles UI and orchestrates calls to other modules.
- `sidebar.py`: Defines the sidebar UI components, including file upload.
- `data_ingestion.py`: Handles ingestion of raw JSON data, processing, and storage into the SQLite database.
- `data_analysis.py`: Performs data loading from the database, calculations, and analytical functions (summaries, period costs, sentiment, keywords, forecasting).
- `visualization.py`: Responsible for generating and displaying various charts and visual elements based on data from `DataAnalysis`.
- **requirements.txt**: A list of all Python dependencies required for the project.
- **data/**: Directory to store the SQLite database.

## Key Functions

- **`DataIngestion` (`data_ingestion.py`):**
    - `process_json_to_sqlite()`: Processes uploaded JSON data, calculates initial metrics (like tokens and costs), and stores the structured data in an SQLite database. Includes helper methods for token counting, date conversion, and initial data processing.
- **`DataAnalysis` (`data_analysis.py`):**
    - `load_conversations_data()`: Loads detailed conversation and message data from the SQLite database.
    - `calculate_conversation_summary()`: Aggregates data to provide a summary for each conversation (e.g., total messages, tokens, cost).
    - `calculate_period_costs()`: Calculates total costs and token usage, grouped by specified time periods (daily, weekly, monthly).
    - `perform_sentiment_analysis()`: Analyzes message content to determine sentiment (positive, negative, neutral).
    - `perform_keyword_analysis()`: Identifies and counts frequently used keywords in messages.
    - `perform_cost_forecasting()`: Forecasts future costs based on historical data using linear regression.
    - `calculate_token_efficiency()`: Calculates the ratio of output tokens to input tokens.
- **`Visualization` (`visualization.py`):**
    - Contains methods like `display_summary_statistics()`, `display_period_breakdown()`, `display_sentiment_analysis()`, `display_keyword_analysis()`, `display_cost_forecasting()`, and `display_token_efficiency()` which are responsible for rendering various UI components and Plotly charts in the Streamlit dashboard, using data prepared by the `DataAnalysis` module.
- **`sidebar.py`:**
    - The `sidebar()` function renders the Streamlit sidebar, including instructions and the file uploader widget. It uses `DataIngestion` to process the uploaded file.
- **`dashboard.py`:**
    - The `main_dashboard()` function sets up the main page structure with tabs for "Conversations", "Statistics", and "Advanced Analytics". It integrates functionalities from `DataAnalysis` and `Visualization` to display data and charts. The `main()` function in this file initializes the application.

## Future Work

- **Sentiment Analysis**: Implement a more sophisticated sentiment analysis using NLP models.
- **Keyword Analysis**: Enhance keyword analysis with better text processing techniques.
- **Cost Forecasting**: Improve the forecasting model for more accurate predictions.
- **Interactive Visualizations**: Add more interactive and customizable visualizations.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the OpenAI team for developing such an amazing tool and to the Streamlit community for their continuous support and improvements to the platform.

---

Feel free to reach out with any questions or feedback. Happy analyzing!

---

**Author**: Thiago MadPin
**Contact**: [madpin@gmail.com]  