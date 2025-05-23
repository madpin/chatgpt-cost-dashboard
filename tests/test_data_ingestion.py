import unittest
import pandas as pd
from data_ingestion import DataIngestion # Assuming data_ingestion.py is in the parent directory or PYTHONPATH is set

class TestDataIngestion(unittest.TestCase):

    def setUp(self):
        self.ingestor = DataIngestion()
        # Mock DB_FILE if necessary for some tests, though these initial ones might not need it
        # self.ingestor.db_file = ":memory:" 

    def test_count_tokens(self):
        self.assertEqual(self.ingestor.count_tokens("Hello world"), 2)
        self.assertEqual(self.ingestor.count_tokens(""), 0)
        # Add more test cases if specific model behavior is known

    def test_safe_to_datetime(self):
        self.assertEqual(self.ingestor.safe_to_datetime(1678886400.0), "2023-03-15 13:20:00") # Example timestamp
        self.assertIsNone(self.ingestor.safe_to_datetime(None))
        self.assertIsNone(self.ingestor.safe_to_datetime("invalid")) 
        self.assertIsNone(self.ingestor.safe_to_datetime(1e19)) # Overflow

    def test_calculate_costs(self):
        data = {
            'input_tokens': [1000000, 2000000],
            'output_tokens': [500000, 1000000]
        }
        df = pd.DataFrame(data)
        # Assuming INPUT_COST_PER_M = 3 and OUTPUT_COST_PER_M = 15 (as defined in data_ingestion.py)
        # These might need to be accessed or mocked if they change
        # For now, using the constants: INPUT_COST_PER_M = 3, OUTPUT_COST_PER_M = 15
        
        # Calculation:
        # input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_M
        # output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
        # total_cost = input_cost + output_cost

        # Row 1:
        # input_cost_1 = (1000000 / 1000000) * 3 = 3
        # output_cost_1 = (500000 / 1000000) * 15 = 7.5
        # total_cost_1 = 3 + 7.5 = 10.5

        # Row 2:
        # input_cost_2 = (2000000 / 1000000) * 3 = 6
        # output_cost_2 = (1000000 / 1000000) * 15 = 15
        # total_cost_2 = 6 + 15 = 21

        expected_input_costs = [3.0, 6.0]
        expected_output_costs = [7.5, 15.0]
        expected_total_costs = [10.5, 21.0]

        result_df = self.ingestor.calculate_costs(df.copy()) # Use .copy() to avoid modifying original df if calculate_costs modifies in-place

        pd.testing.assert_series_equal(result_df['input_cost'], pd.Series(expected_input_costs, name='input_cost'), check_dtype=False)
        pd.testing.assert_series_equal(result_df['output_cost'], pd.Series(expected_output_costs, name='output_cost'), check_dtype=False)
        pd.testing.assert_series_equal(result_df['total_cost'], pd.Series(expected_total_costs, name='total_cost'), check_dtype=False)

if __name__ == '__main__':
    unittest.main()
