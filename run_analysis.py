import os
import argparse
from sensortest import SensorDataAnalyzer


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze sensor and food rating data.')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base directory containing both sensor and food rating data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')

    args = parser.parse_args()

    # Create analyzer
    analyzer = SensorDataAnalyzer(
        base_folder=args.base_dir,
        output_dir=args.output_dir
    )

    # Run analysis
    print("\n" + "=" * 70)
    print("Starting data analysis")
    print("=" * 70)

    success = analyzer.analyze_all_data()

    if success:
        print("\n" + "=" * 70)
        print("Analysis completed successfully!")
        print(f"Results saved to: {os.path.abspath(analyzer.output_dir)}")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("Analysis completed with errors. Check the log file for details.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    main()