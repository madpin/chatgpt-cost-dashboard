import unittest
import pandas as pd
from data_analysis import DataAnalysis # Assuming data_analysis.py is in the parent directory or PYTHONPATH is set

class TestDataAnalysis(unittest.TestCase):

    def setUp(self):
        self.analyzer = DataAnalysis()
        # Mock DB_FILE for DataAnalysis as it tries to connect in __init__ or methods
        # For these specific tests, we might not hit the DB if we pass DataFrames directly
        # However, load_conversations_data etc. would need a mock DB.
        # For now, these tests will focus on methods that take DataFrames as input.
        self.analyzer.db_file = ":memory:" # Use in-memory SQLite for tests if DB interaction is unavoidable

    def test_calculate_token_efficiency(self):
        data_efficient = {'input_tokens': [100, 200], 'output_tokens': [80, 160]} # 80% efficiency
        df_efficient = pd.DataFrame(data_efficient)
        self.assertAlmostEqual(self.analyzer.calculate_token_efficiency(df_efficient), 80.0)

        data_inefficient = {'input_tokens': [100], 'output_tokens': [20]} # 20% efficiency
        df_inefficient = pd.DataFrame(data_inefficient)
        self.assertAlmostEqual(self.analyzer.calculate_token_efficiency(df_inefficient), 20.0)
        
        data_zero_input = {'input_tokens': [0], 'output_tokens': [100]}
        df_zero_input = pd.DataFrame(data_zero_input)
        self.assertEqual(self.analyzer.calculate_token_efficiency(df_zero_input), 0.0) # Avoid division by zero

        data_empty = {'input_tokens': [], 'output_tokens': []}
        df_empty = pd.DataFrame(data_empty)
        # Depending on implementation, this might be 0.0 or raise an error.
        # The current implementation of calculate_token_efficiency handles empty sum by returning 0.0.
        self.assertEqual(self.analyzer.calculate_token_efficiency(df_empty), 0.0)


    def test_perform_sentiment_analysis(self):
        data = {'message_content': ["This is great", "This is bad", "This is neutral", None]}
        df = pd.DataFrame(data)
        result_df = self.analyzer.perform_sentiment_analysis(df.copy())
        expected_sentiments = pd.Series(["positive", "negative", "neutral", "neutral"], name="sentiment")
        pd.testing.assert_series_equal(result_df['sentiment'], expected_sentiments, check_dtype=False)

    def test_perform_keyword_analysis(self):
        data = {'message_content': ["apple banana apple", "banana orange", "grape"]}
        df = pd.DataFrame(data)
        result_df = self.analyzer.perform_keyword_analysis(df) # This should return top N, default 20
        
        # Expected counts: apple: 2, banana: 2, orange: 1, grape: 1
        # Order might vary for ties if not explicitly handled, but top keywords should be present
        self.assertIn('apple', result_df['keyword'].values)
        self.assertIn('banana', result_df['keyword'].values)
        
        apple_freq = result_df[result_df['keyword'] == 'apple']['frequency'].iloc[0]
        banana_freq = result_df[result_df['keyword'] == 'banana']['frequency'].iloc[0]
        
        self.assertEqual(apple_freq, 2)
        self.assertEqual(banana_freq, 2)
        self.assertTrue(len(result_df) <= 20)


if __name__ == '__main__':
    unittest.main()
