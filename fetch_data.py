import sys
import os

# Ensure data_pipeline is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_pipeline'))

from data_pipeline.fetch_predictive_data import fetch_race_data
from data_pipeline.clean_predictive_data import main as pipeline_main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch raw data')
    parser.add_argument('--process-only', action='store_true', help='Only process existing raw data')
    args = parser.parse_args()

    if args.fetch_only:
        fetch_race_data([2022, 2023, 2024], max_rounds_per_year=4)
    elif args.process_only:
        pipeline_main()
    else:
        # Full pipeline
        fetch_race_data([2022, 2023, 2024], max_rounds_per_year=4)
        pipeline_main()
