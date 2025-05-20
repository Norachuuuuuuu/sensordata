import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import logging
import traceback
import time

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SensorDataAnalyzer")


class SensorDataAnalyzer:
    """
    Main class for analyzing both sensor data and food rating experiment data
    """

    def __init__(self, base_folder=None, output_dir=None):
        """
        Initialize the analyzer with the base folder containing data

        Parameters:
        -----------
        base_folder : str
            Path to the base folder containing sensor data and food rating folders
        output_dir : str
            Path to the output directory
        """
        # Auto-detect the base folder if not provided
        if base_folder is None:
            self.base_folder = self._auto_detect_base_folder()
        else:
            self.base_folder = base_folder

        # Default output directory
        self.output_dir = output_dir if output_dir else "analysis_results"

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Sampling frequency for ITUG/ISWAY data (Hz)
        self.sampling_freq = 100.0

        # Initialize data structures
        self.summary_data = []

        # Get the paths to different data folders
        self.sensor_data_folder = self._find_sensor_data_folder()
        self.food_rating_folder = self._find_food_rating_folder()

        # Print detected folders
        logger.info("Base folder: %s", os.path.abspath(self.base_folder))
        logger.info("Output directory: %s", os.path.abspath(self.output_dir))
        logger.info("Sensor data folder: %s",
                    os.path.abspath(self.sensor_data_folder) if self.sensor_data_folder else "Not found")
        logger.info("Food rating folder: %s",
                    os.path.abspath(self.food_rating_folder) if self.food_rating_folder else "Not found")

    def _auto_detect_base_folder(self):
        """
        Auto-detect the base folder based on current directory structure
        """
        # Current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Check if we're already in the project directory
        if os.path.exists(os.path.join(current_dir, "ITUG-ISWAY")) or \
                os.path.exists(os.path.join(current_dir, "foodratingexperimentresults")):
            return current_dir

        # Check parent directory
        parent_dir = os.path.dirname(current_dir)
        if os.path.exists(os.path.join(parent_dir, "ITUG-ISWAY")) or \
                os.path.exists(os.path.join(parent_dir, "foodratingexperimentresults")):
            return parent_dir

        # Check subdirectories of current directory
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "ITUG-ISWAY")) or \
                        os.path.exists(os.path.join(item_path, "foodratingexperimentresults")):
                    return item_path

        # If no folder is found, default to current directory
        logger.warning("Could not auto-detect project folder, using current directory")
        return current_dir

    def _find_sensor_data_folder(self):
        """
        Find the sensor data folder (ITUG-ISWAY)
        """
        # Check direct path
        direct_path = os.path.join(self.base_folder, "ITUG-ISWAY")
        if os.path.exists(direct_path) and os.path.isdir(direct_path):
            return direct_path

        # Check for a folder containing sensor data files
        for item in os.listdir(self.base_folder):
            item_path = os.path.join(self.base_folder, item)
            if os.path.isdir(item_path):
                # Check if folder contains any ITUG or ISWAY files
                for root, _, files in os.walk(item_path):
                    if any("ITUG" in f.upper() or "ISWAY" in f.upper() for f in files):
                        return item_path

        # If still not found, scan the entire base folder
        for root, _, files in os.walk(self.base_folder):
            if any("ITUG" in f.upper() or "ISWAY" in f.upper() for f in files):
                return os.path.dirname(root)

        # If not found
        logger.warning("Could not find ITUG-ISWAY sensor data folder")
        return None

    def _find_food_rating_folder(self):
        """
        Find the food rating experiment results folder
        """
        # Check direct path
        direct_path = os.path.join(self.base_folder, "foodratingexperimentresults")
        if os.path.exists(direct_path) and os.path.isdir(direct_path):
            return direct_path

        # Check for a folder that might contain rating JSON files
        for item in os.listdir(self.base_folder):
            item_path = os.path.join(self.base_folder, item)
            if os.path.isdir(item_path) and "food" in item.lower():
                return item_path

        # Scan the base folder for JSON files with a pattern common to rating files
        for root, _, files in os.walk(self.base_folder):
            if any(f.endswith(".json") for f in files):
                # Check inside the JSON files for rating-related keys
                for file in files:
                    if file.endswith(".json"):
                        try:
                            with open(os.path.join(root, file), 'r') as f:
                                data = json.load(f)
                                # Check if the JSON has rating-related keys
                                if any(key in data for key in
                                       ["initialRatings", "chosenPairs", "rejectedPairs", "finalRatings"]):
                                    return root
                        except:
                            continue

        # If not found
        logger.warning("Could not find food rating experiment results folder")
        return None

    def analyze_all_data(self):
        """
        Main method to analyze all participant data
        """
        logger.info("=" * 50)
        logger.info("Starting analysis of all data")
        logger.info("=" * 50)

        # Create output directories
        sensor_output_dir = os.path.join(self.output_dir, "sensor_analysis")
        food_output_dir = os.path.join(self.output_dir, "food_analysis")
        os.makedirs(sensor_output_dir, exist_ok=True)
        os.makedirs(food_output_dir, exist_ok=True)

        # Analyze sensor data if available
        if self.sensor_data_folder:
            logger.info("Analyzing sensor data...")
            self.analyze_all_sensor_data(sensor_output_dir)
        else:
            logger.warning("Skipping sensor data analysis - folder not found")

        # Analyze food rating data if available
        if self.food_rating_folder:
            logger.info("Analyzing food rating data...")
            self.analyze_all_food_ratings(food_output_dir)
        else:
            logger.warning("Skipping food rating analysis - folder not found")

        # Generate consolidated summary
        self.generate_consolidated_summary()

        logger.info("=" * 50)
        logger.info("Analysis completed")
        logger.info("=" * 50)

        return True

    def analyze_all_sensor_data(self, output_dir):
        """
        Analyze all sensor data (ITUG and ISWAY tests)
        """
        # Track start time for performance metrics
        start_time = time.time()

        try:
            # Find all sensor data files
            logger.info("Scanning for sensor data files...")

            # Method 1: Check if files are organized by participant folders
            participant_folders = self._find_sensor_participant_folders()

            if participant_folders:
                # Process by participant folders
                logger.info(f"Found {len(participant_folders)} participant folders")

                # Process each participant folder
                for i, (folder_name, folder_path) in enumerate(participant_folders.items()):
                    logger.info(
                        f"Processing sensor data for participant {i + 1}/{len(participant_folders)}: {folder_name}")
                    self.process_participant_sensor_data(folder_name, folder_path, output_dir)
            else:
                # Method 2: Files not in participant folders, but in a flat structure
                all_sensor_files = self._find_all_sensor_files()

                if all_sensor_files:
                    # Group files by participant
                    participant_files = self._group_sensor_files_by_participant(all_sensor_files)

                    logger.info(f"Found {len(participant_files)} participants in sensor data")

                    # Process each participant's files
                    for i, (participant_name, files) in enumerate(participant_files.items()):
                        logger.info(
                            f"Processing sensor data for participant {i + 1}/{len(participant_files)}: {participant_name}")
                        self.process_participant_sensor_files(participant_name, files, output_dir)
                else:
                    logger.error("No sensor data files found!")
                    return False

            # Generate aggregate visualizations
            self.generate_sensor_aggregate_visualizations(output_dir)

            # Print performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Sensor data analysis completed in {elapsed_time:.2f} seconds")

            return True

        except Exception as e:
            logger.error(f"Error analyzing sensor data: {e}")
            logger.error(traceback.format_exc())
            return False

    def _find_sensor_participant_folders(self):
        """
        Find participant folders in the sensor data folder
        """
        participant_folders = {}

        # Check if sensor_data_folder exists and is a directory
        if not self.sensor_data_folder or not os.path.isdir(self.sensor_data_folder):
            return participant_folders

        # Look for participant folders
        for item in os.listdir(self.sensor_data_folder):
            item_path = os.path.join(self.sensor_data_folder, item)
            if os.path.isdir(item_path):
                # Check if this folder contains ITUG or ISWAY files
                for root, _, files in os.walk(item_path):
                    if any("ITUG" in f.upper() or "ISWAY" in f.upper() for f in files):
                        participant_folders[item] = item_path
                        break

        return participant_folders

    def _find_all_sensor_files(self):
        """
        Find all sensor data files in the folder structure
        """
        all_files = []

        # Check if sensor_data_folder exists and is a directory
        if not self.sensor_data_folder or not os.path.isdir(self.sensor_data_folder):
            return all_files

        # Walk through the directory tree
        for root, _, files in os.walk(self.sensor_data_folder):
            for file in files:
                if file.endswith(".csv") and ("ITUG" in file.upper() or "ISWAY" in file.upper()):
                    all_files.append(os.path.join(root, file))

        return all_files

    def _group_sensor_files_by_participant(self, all_files):
        """
        Group sensor files by participant
        """
        participant_files = defaultdict(list)

        for file_path in all_files:
            file_name = os.path.basename(file_path)

            # Try to extract participant name from the file name
            # Expected format: Name-Gender-Age-Weight-Height-TESTTYPE-PHASE.csv
            # Example: AhmetLutfullah-E-19-83-183-ITUG-PRE.csv

            # Split by dash and look for participant identifiers
            parts = file_name.split('-')

            if len(parts) >= 5:
                # Check if the expected parts match the pattern (Gender is E/K, Age/Weight/Height are numbers)
                if parts[1] in ['E', 'K'] and parts[2].isdigit() and parts[3].isdigit() and parts[4].split('.')[
                    0].isdigit():
                    # Extract participant name (first part)
                    participant_name = parts[0]

                    # Build full participant identifier
                    participant_id = f"{participant_name}-{parts[1]}-{parts[2]}-{parts[3]}-{parts[4].split('.')[0]}"

                    participant_files[participant_id].append(file_path)
                else:
                    # If not matching expected pattern, use a fallback approach
                    # Just use the first part as participant name
                    participant_files[parts[0]].append(file_path)
            else:
                # Fallback: use file name without extension as participant identifier
                participant_name = os.path.splitext(file_name)[0]
                participant_files[participant_name].append(file_path)

        return participant_files

    def process_participant_sensor_data(self, participant_name, folder_path, output_dir):
        """
        Process sensor data for a participant with data in a dedicated folder
        """
        try:
            # Create participant output directory
            participant_output_dir = os.path.join(output_dir, participant_name)
            os.makedirs(participant_output_dir, exist_ok=True)

            # Parse participant information from folder name
            participant_info = self.parse_participant_info(participant_name)

            if not participant_info:
                logger.warning(f"Could not parse participant info from folder name: {participant_name}")
                return

            logger.info(f"Processing data for {participant_info.get('name', 'Unknown')}, "
                        f"{participant_info.get('gender', 'Unknown')}, "
                        f"{participant_info.get('age', 'Unknown')} years, "
                        f"{participant_info.get('weight', 'Unknown')} kg, "
                        f"{participant_info.get('height', 'Unknown')} cm")

            # Find all sensor data files in the participant folder
            all_files = self.list_data_files(folder_path)

            # Categorize files by test type (pre/post ITUG/ISWAY)
            test_files = self.categorize_test_files(all_files, folder_path)

            # Process each test type
            test_results = {}
            for test_type, file_path in test_files.items():
                if file_path:
                    logger.info(f"Processing {test_type} test")
                    test_output_dir = os.path.join(participant_output_dir, test_type)
                    os.makedirs(test_output_dir, exist_ok=True)

                    # Process the test file
                    result = self.process_test_file(file_path, test_output_dir)
                    test_results[test_type] = result

            # Load food rating data if available
            ratings_data = self.load_ratings_data_for_participant(participant_name)

            # Combine all data into a participant summary
            participant_summary = {
                **participant_info,
                **{f"{test_type}_metrics": results for test_type, results in test_results.items()},
                **ratings_data
            }

            # Save participant summary
            summary_file = os.path.join(participant_output_dir, "participant_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(participant_summary, f, indent=2)

            # Add to summary data
            self.add_to_summary_data(participant_summary)

            logger.info(f"Completed processing for {participant_info.get('name', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error processing participant {participant_name}: {e}")
            logger.error(traceback.format_exc())
            return False

    def process_participant_sensor_files(self, participant_name, files, output_dir):
        """
        Process sensor files for a participant (when files are in a flat structure)
        """
        try:
            # Create participant output directory
            participant_output_dir = os.path.join(output_dir, participant_name)
            os.makedirs(participant_output_dir, exist_ok=True)

            # Parse participant information from name
            participant_info = self.parse_participant_info(participant_name)

            if not participant_info:
                logger.warning(f"Could not parse participant info from name: {participant_name}")
                return

            logger.info(f"Processing data for {participant_info.get('name', 'Unknown')}, "
                        f"{participant_info.get('gender', 'Unknown')}, "
                        f"{participant_info.get('age', 'Unknown')} years, "
                        f"{participant_info.get('weight', 'Unknown')} kg, "
                        f"{participant_info.get('height', 'Unknown')} cm")

            # Categorize files by test type
            test_files = self.categorize_test_files_from_list(files)

            # Process each test type
            test_results = {}
            for test_type, file_path in test_files.items():
                if file_path:
                    logger.info(f"Processing {test_type} test")
                    test_output_dir = os.path.join(participant_output_dir, test_type)
                    os.makedirs(test_output_dir, exist_ok=True)

                    # Process the test file
                    result = self.process_test_file(file_path, test_output_dir)
                    test_results[test_type] = result

            # Load food rating data if available
            ratings_data = self.load_ratings_data_for_participant(participant_name)

            # Combine all data into a participant summary
            participant_summary = {
                **participant_info,
                **{f"{test_type}_metrics": results for test_type, results in test_results.items()},
                **ratings_data
            }

            # Save participant summary
            summary_file = os.path.join(participant_output_dir, "participant_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(participant_summary, f, indent=2)

            # Add to summary data
            self.add_to_summary_data(participant_summary)

            logger.info(f"Completed processing for {participant_info.get('name', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error processing participant files for {participant_name}: {e}")
            logger.error(traceback.format_exc())
            return False

    def categorize_test_files_from_list(self, files):
        """
        Categorize files by test type from a list of file paths
        """
        test_files = {
            "pre_itug": None,
            "pre_isway": None,
            "post_itug": None,
            "post_isway": None
        }

        for file_path in files:
            file_name = os.path.basename(file_path).lower()

            if "pre" in file_name and "itug" in file_name:
                test_files["pre_itug"] = file_path
            elif "pre" in file_name and "isway" in file_name:
                test_files["pre_isway"] = file_path
            elif "post" in file_name and "itug" in file_name:
                test_files["post_itug"] = file_path
            elif "post" in file_name and "isway" in file_name:
                test_files["post_isway"] = file_path

        # Print summary of categorized files
        for test_type, file_path in test_files.items():
            if file_path:
                logger.info(f"  {test_type}: {os.path.basename(file_path)}")
            else:
                logger.info(f"  {test_type}: Not found")

        return test_files

    def load_ratings_data_for_participant(self, participant_name):
        """
        Load food rating data for a specific participant
        """
        ratings_data = {
            "initial_rating": None,
            "final_rating": None,
            "rejected_pair_rating_diff": None,
            "chosen_pair_rating_diff": None,
            "computer_pair_rating_diff": None,
            "timeout_count": None
        }

        if not self.food_rating_folder:
            return ratings_data

        try:
            # Look for a JSON file with the participant's name
            participant_rating_file = None

            for file in os.listdir(self.food_rating_folder):
                if file.endswith(".json") and participant_name in file:
                    participant_rating_file = os.path.join(self.food_rating_folder, file)
                    break

            if not participant_rating_file:
                logger.warning(f"No food rating data found for participant: {participant_name}")
                return ratings_data

            # Load the JSON file
            with open(participant_rating_file, 'r') as f:
                data = json.load(f)

            # Extract rating data
            if "initialRatings" in data and "finalRatings" in data:
                # Calculate average ratings
                if data["initialRatings"]:
                    ratings_data["initial_rating"] = sum(data["initialRatings"].values()) / len(data["initialRatings"])

                if data["finalRatings"]:
                    ratings_data["final_rating"] = sum(data["finalRatings"].values()) / len(data["finalRatings"])

                # Calculate average rating differences for chosen and rejected pairs
                if "chosenPairs" in data and data["chosenPairs"]:
                    chosen_diffs = []
                    for item in data["chosenPairs"]:
                        if item in data["finalRatings"] and item in data["initialRatings"]:
                            chosen_diffs.append(data["finalRatings"][item] - data["initialRatings"][item])

                    if chosen_diffs:
                        ratings_data["chosen_pair_rating_diff"] = sum(chosen_diffs) / len(chosen_diffs)

                if "rejectedPairs" in data and data["rejectedPairs"]:
                    rejected_diffs = []
                    for item in data["rejectedPairs"]:
                        if item in data["finalRatings"] and item in data["initialRatings"]:
                            rejected_diffs.append(data["finalRatings"][item] - data["initialRatings"][item])

                    if rejected_diffs:
                        ratings_data["rejected_pair_rating_diff"] = sum(rejected_diffs) / len(rejected_diffs)

                if "computerChosenPairs" in data and data["computerChosenPairs"]:
                    computer_diffs = []
                    for item in data["computerChosenPairs"]:
                        if item in data["finalRatings"] and item in data["initialRatings"]:
                            computer_diffs.append(data["finalRatings"][item] - data["initialRatings"][item])

                    if computer_diffs:
                        ratings_data["computer_pair_rating_diff"] = sum(computer_diffs) / len(computer_diffs)

            # Get timeout count if available
            if "timeoutCount" in data:
                ratings_data["timeout_count"] = data["timeoutCount"]

            logger.info(f"Loaded food rating data for participant: {participant_name}")
            return ratings_data

        except Exception as e:
            logger.error(f"Error loading food rating data for {participant_name}: {e}")
            return ratings_data

    def analyze_all_food_ratings(self, output_dir):
        """
        Analyze all food rating data
        """
        start_time = time.time()

        try:
            if not self.food_rating_folder or not os.path.isdir(self.food_rating_folder):
                logger.error("Food rating folder not found or not a directory")
                return False

            # Get all JSON files in the food rating folder
            json_files = [f for f in os.listdir(self.food_rating_folder)
                          if f.endswith(".json") and os.path.isfile(os.path.join(self.food_rating_folder, f))]

            if not json_files:
                logger.error("No JSON files found in food rating folder")
                return False

            logger.info(f"Found {len(json_files)} food rating data files")

            # Process each JSON file
            food_data = []

            for i, json_file in enumerate(json_files):
                logger.info(f"Processing food rating data {i + 1}/{len(json_files)}: {json_file}")

                file_path = os.path.join(self.food_rating_folder, json_file)

                try:
                    # Parse participant name from file name
                    participant_name = os.path.splitext(json_file)[0]
                    participant_info = self.parse_participant_info(participant_name)

                    # Load the JSON file
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Process the food rating data
                    processed_data = self.process_food_rating_data(data, participant_info)

                    # Add to food data list
                    food_data.append(processed_data)

                    # Save individual processed data
                    participant_output_dir = os.path.join(output_dir, participant_name)
                    os.makedirs(participant_output_dir, exist_ok=True)

                    with open(os.path.join(participant_output_dir, "processed_food_data.json"), 'w') as f:
                        json.dump(processed_data, f, indent=2)

                except Exception as e:
                    logger.error(f"Error processing food rating file {json_file}: {e}")
                    logger.error(traceback.format_exc())

            # Generate food rating visualizations
            if food_data:
                self.generate_food_rating_visualizations(food_data, output_dir)

            # Save combined food data
            with open(os.path.join(output_dir, "all_food_ratings.json"), 'w') as f:
                json.dump(food_data, f, indent=2)

            # Print performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Food rating analysis completed in {elapsed_time:.2f} seconds")

            return True

        except Exception as e:
            logger.error(f"Error analyzing food rating data: {e}")
            logger.error(traceback.format_exc())
            return False

    def process_food_rating_data(self, data, participant_info):
        """
        Process food rating data for a participant
        """
        processed_data = {
            "participant_info": participant_info,
            "summary": {},
            "ratings": {},
            "chosen_pairs": [],
            "rejected_pairs": [],
            "computer_chosen_pairs": []
        }

        try:
            # Extract and process initial and final ratings
            initial_ratings = data.get("initialRatings", {})
            final_ratings = data.get("finalRatings", {})

            # Calculate average ratings
            if initial_ratings:
                avg_initial = sum(initial_ratings.values()) / len(initial_ratings)
                processed_data["summary"]["average_initial_rating"] = avg_initial

            if final_ratings:
                avg_final = sum(final_ratings.values()) / len(final_ratings)
                processed_data["summary"]["average_final_rating"] = avg_final

            # Calculate rating differences
            rating_diffs = {}
            for item, initial in initial_ratings.items():
                if item in final_ratings:
                    rating_diffs[item] = final_ratings[item] - initial

            # Store ratings data
            processed_data["ratings"]["initial"] = initial_ratings
            processed_data["ratings"]["final"] = final_ratings
            processed_data["ratings"]["differences"] = rating_diffs

            # Process chosen and rejected pairs
            chosen_pairs = data.get("chosenPairs", [])
            rejected_pairs = data.get("rejectedPairs", [])
            computer_chosen_pairs = data.get("computerChosenPairs", [])

            # Extract item names and their rating differences
            chosen_pair_diffs = []
            for item in chosen_pairs:
                if item in rating_diffs:
                    chosen_pair_diffs.append({
                        "item": item,
                        "difference": rating_diffs[item]
                    })

            rejected_pair_diffs = []
            for item in rejected_pairs:
                if item in rating_diffs:
                    rejected_pair_diffs.append({
                        "item": item,
                        "difference": rating_diffs[item]
                    })

            computer_pair_diffs = []
            for item in computer_chosen_pairs:
                if item in rating_diffs:
                    computer_pair_diffs.append({
                        "item": item,
                        "difference": rating_diffs[item]
                    })

            # Store processed pairs data
            processed_data["chosen_pairs"] = chosen_pair_diffs
            processed_data["rejected_pairs"] = rejected_pair_diffs
            processed_data["computer_chosen_pairs"] = computer_pair_diffs

            # Calculate average differences
            if chosen_pair_diffs:
                avg_chosen_diff = sum(item["difference"] for item in chosen_pair_diffs) / len(chosen_pair_diffs)
                processed_data["summary"]["average_chosen_difference"] = avg_chosen_diff

            if rejected_pair_diffs:
                avg_rejected_diff = sum(item["difference"] for item in rejected_pair_diffs) / len(rejected_pair_diffs)
                processed_data["summary"]["average_rejected_difference"] = avg_rejected_diff

            if computer_pair_diffs:
                avg_computer_diff = sum(item["difference"] for item in computer_pair_diffs) / len(computer_pair_diffs)
                processed_data["summary"]["average_computer_difference"] = avg_computer_diff

            # Add timeout count if available
            if "timeoutCount" in data:
                processed_data["summary"]["timeout_count"] = data["timeoutCount"]

            return processed_data

        except Exception as e:
            logger.error(f"Error processing food rating data: {e}")
            logger.error(traceback.format_exc())
            return processed_data

    def generate_food_rating_visualizations(self, food_data, output_dir):
        """
        Generate visualizations for food rating data
        """
        try:
            # Create visualization directory
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # Extract data for visualizations
            avg_initial_ratings = []
            avg_final_ratings = []
            avg_chosen_diffs = []
            avg_rejected_diffs = []
            avg_computer_diffs = []

            male_initial = []
            male_final = []
            female_initial = []
            female_final = []

            for participant_data in food_data:
                summary = participant_data.get("summary", {})

                # Collect average ratings
                if "average_initial_rating" in summary:
                    avg_initial_ratings.append(summary["average_initial_rating"])

                if "average_final_rating" in summary:
                    avg_final_ratings.append(summary["average_final_rating"])

                # Collect average differences
                if "average_chosen_difference" in summary:
                    avg_chosen_diffs.append(summary["average_chosen_difference"])

                if "average_rejected_difference" in summary:
                    avg_rejected_diffs.append(summary["average_rejected_difference"])

                if "average_computer_difference" in summary:
                    avg_computer_diffs.append(summary["average_computer_difference"])

                # Collect gender-specific data
                participant_info = participant_data.get("participant_info", {})
                gender = participant_info.get("gender", "Unknown")

                if gender == "Male" and "average_initial_rating" in summary:
                    male_initial.append(summary["average_initial_rating"])

                if gender == "Male" and "average_final_rating" in summary:
                    male_final.append(summary["average_final_rating"])

                if gender == "Female" and "average_initial_rating" in summary:
                    female_initial.append(summary["average_initial_rating"])

                if gender == "Female" and "average_final_rating" in summary:
                    female_final.append(summary["average_final_rating"])

            # 1. Initial vs Final Ratings Distribution
            plt.figure(figsize=(10, 6))

            if avg_initial_ratings and avg_final_ratings:
                plt.boxplot([avg_initial_ratings, avg_final_ratings], labels=['Initial Ratings', 'Final Ratings'])
                plt.ylabel('Average Rating')
                plt.title('Distribution of Initial vs Final Average Ratings')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(viz_dir, 'initial_vs_final_ratings.png'))

            # 2. Rating Differences by Pair Type
            plt.figure(figsize=(10, 6))

            data_to_plot = []
            labels = []

            if avg_chosen_diffs:
                data_to_plot.append(avg_chosen_diffs)
                labels.append('Chosen Pairs')

            if avg_rejected_diffs:
                data_to_plot.append(avg_rejected_diffs)
                labels.append('Rejected Pairs')

            if avg_computer_diffs:
                data_to_plot.append(avg_computer_diffs)
                labels.append('Computer Pairs')

            if data_to_plot:
                plt.boxplot(data_to_plot, labels=labels)
                plt.ylabel('Average Rating Difference')
                plt.title('Rating Differences by Pair Type')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(viz_dir, 'rating_differences_by_pair.png'))

            # 3. Gender Comparison
            plt.figure(figsize=(10, 6))

            width = 0.35
            x = np.arange(2)

            if male_initial and male_final and female_initial and female_final:
                male_means = [np.mean(male_initial), np.mean(male_final)]
                female_means = [np.mean(female_initial), np.mean(female_final)]

                plt.bar(x - width / 2, male_means, width, label='Male')
                plt.bar(x + width / 2, female_means, width, label='Female')

                plt.ylabel('Average Rating')
                plt.title('Initial and Final Ratings by Gender')
                plt.xticks(x, ['Initial', 'Final'])
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(viz_dir, 'ratings_by_gender.png'))

            plt.close('all')
            logger.info(f"Generated food rating visualizations in {viz_dir}")

        except Exception as e:
            logger.error(f"Error generating food rating visualizations: {e}")
            logger.error(traceback.format_exc())

    def generate_consolidated_summary(self):
        """
        Generate a consolidated summary of all analysis results
        """
        try:
            # Create a comprehensive report
            report = {
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sensor_data_summary": self.generate_sensor_data_summary(),
                "food_rating_summary": self.generate_food_rating_summary()
            }

            # Save to JSON
            report_file = os.path.join(self.output_dir, "consolidated_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Generated consolidated report: {report_file}")

            # Create a text summary
            text_report = self.generate_text_report(report)

            # Save to text file
            text_file = os.path.join(self.output_dir, "analysis_summary.txt")
            with open(text_file, 'w') as f:
                f.write(text_report)

            logger.info(f"Generated text summary: {text_file}")

        except Exception as e:
            logger.error(f"Error generating consolidated summary: {e}")
            logger.error(traceback.format_exc())

    def generate_sensor_data_summary(self):
        """
        Generate a summary of sensor data analysis
        """
        summary = {
            "participants_analyzed": len(self.summary_data),
            "demographics": {},
            "test_completion": {},
            "metrics": {}
        }

        if not self.summary_data:
            return summary

        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(self.summary_data)

            # Demographics
            if 'gender' in df.columns:
                gender_counts = df['gender'].value_counts().to_dict()
                summary["demographics"]["gender_distribution"] = gender_counts

            if 'age' in df.columns:
                age_data = df['age'].dropna()
                if len(age_data) > 0:
                    summary["demographics"]["age"] = {
                        "min": float(age_data.min()),
                        "max": float(age_data.max()),
                        "mean": float(age_data.mean()),
                        "median": float(age_data.median())
                    }

            if 'weight' in df.columns:
                weight_data = df['weight'].dropna()
                if len(weight_data) > 0:
                    summary["demographics"]["weight_kg"] = {
                        "min": float(weight_data.min()),
                        "max": float(weight_data.max()),
                        "mean": float(weight_data.mean()),
                        "median": float(weight_data.median())
                    }

            if 'height' in df.columns:
                height_data = df['height'].dropna()
                if len(height_data) > 0:
                    summary["demographics"]["height_cm"] = {
                        "min": float(height_data.min()),
                        "max": float(height_data.max()),
                        "mean": float(height_data.mean()),
                        "median": float(height_data.median())
                    }

            # Test completion rates
            test_types = ["pre_itug", "pre_isway", "post_itug", "post_isway"]

            for test_type in test_types:
                col_name = f"{test_type}_duration"
                if col_name in df.columns:
                    complete_count = df[col_name].notna().sum()
                    summary["test_completion"][test_type] = {
                        "completed": int(complete_count),
                        "completion_rate": float(complete_count / len(df) * 100)
                    }

            # Key metrics
            # ITUG metrics
            itug_cols = [col for col in df.columns if "itug" in col.lower() and "stride_frequency" in col.lower()]
            for col in itug_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    summary["metrics"][col] = {
                        "mean": float(data.mean()),
                        "median": float(data.median()),
                        "min": float(data.min()),
                        "max": float(data.max())
                    }

            # ISWAY metrics
            isway_cols = [col for col in df.columns if "isway" in col.lower() and "sway_area" in col.lower()]
            for col in isway_cols:
                data = df[col].dropna()
                if len(data) > 0:
                    summary["metrics"][col] = {
                        "mean": float(data.mean()),
                        "median": float(data.median()),
                        "min": float(data.min()),
                        "max": float(data.max())
                    }

            return summary

        except Exception as e:
            logger.error(f"Error generating sensor data summary: {e}")
            logger.error(traceback.format_exc())
            return summary

    def generate_food_rating_summary(self):
        """
        Generate a summary of food rating analysis
        """
        summary = {
            "participants_analyzed": 0,
            "rating_statistics": {},
            "chosen_vs_rejected": {}
        }

        if not self.food_rating_folder:
            return summary

        try:
            # Check if the consolidated food data file exists
            food_data_file = os.path.join(self.output_dir, "food_analysis", "all_food_ratings.json")

            if not os.path.exists(food_data_file):
                return summary

            # Load the food data
            with open(food_data_file, 'r') as f:
                food_data = json.load(f)

            summary["participants_analyzed"] = len(food_data)

            # Calculate statistics
            initial_ratings = []
            final_ratings = []
            chosen_diffs = []
            rejected_diffs = []
            computer_diffs = []

            for participant_data in food_data:
                participant_summary = participant_data.get("summary", {})

                if "average_initial_rating" in participant_summary:
                    initial_ratings.append(participant_summary["average_initial_rating"])

                if "average_final_rating" in participant_summary:
                    final_ratings.append(participant_summary["average_final_rating"])

                if "average_chosen_difference" in participant_summary:
                    chosen_diffs.append(participant_summary["average_chosen_difference"])

                if "average_rejected_difference" in participant_summary:
                    rejected_diffs.append(participant_summary["average_rejected_difference"])

                if "average_computer_difference" in participant_summary:
                    computer_diffs.append(participant_summary["average_computer_difference"])

            # Rating statistics
            if initial_ratings:
                summary["rating_statistics"]["initial_ratings"] = {
                    "mean": float(np.mean(initial_ratings)),
                    "median": float(np.median(initial_ratings)),
                    "min": float(np.min(initial_ratings)),
                    "max": float(np.max(initial_ratings))
                }

            if final_ratings:
                summary["rating_statistics"]["final_ratings"] = {
                    "mean": float(np.mean(final_ratings)),
                    "median": float(np.median(final_ratings)),
                    "min": float(np.min(final_ratings)),
                    "max": float(np.max(final_ratings))
                }

            # Chosen vs Rejected
            if chosen_diffs:
                summary["chosen_vs_rejected"]["chosen_pairs_diff"] = {
                    "mean": float(np.mean(chosen_diffs)),
                    "median": float(np.median(chosen_diffs)),
                    "min": float(np.min(chosen_diffs)),
                    "max": float(np.max(chosen_diffs))
                }

            if rejected_diffs:
                summary["chosen_vs_rejected"]["rejected_pairs_diff"] = {
                    "mean": float(np.mean(rejected_diffs)),
                    "median": float(np.median(rejected_diffs)),
                    "min": float(np.min(rejected_diffs)),
                    "max": float(np.max(rejected_diffs))
                }

            if computer_diffs:
                summary["chosen_vs_rejected"]["computer_pairs_diff"] = {
                    "mean": float(np.mean(computer_diffs)),
                    "median": float(np.median(computer_diffs)),
                    "min": float(np.min(computer_diffs)),
                    "max": float(np.max(computer_diffs))
                }

            return summary

        except Exception as e:
            logger.error(f"Error generating food rating summary: {e}")
            logger.error(traceback.format_exc())
            return summary

    def generate_text_report(self, report):
        """
        Generate a human-readable text report from the consolidated report
        """
        text = "ANALYSIS SUMMARY REPORT\n"
        text += "=" * 80 + "\n\n"
        text += f"Analysis completed on: {report.get('analysis_timestamp', 'Unknown')}\n\n"

        # Sensor Data Summary
        text += "SENSOR DATA ANALYSIS\n"
        text += "-" * 80 + "\n"

        sensor_summary = report.get("sensor_data_summary", {})

        text += f"Participants analyzed: {sensor_summary.get('participants_analyzed', 0)}\n\n"

        # Demographics
        demographics = sensor_summary.get("demographics", {})
        text += "Demographics:\n"

        if "gender_distribution" in demographics:
            text += "  Gender distribution:\n"
            for gender, count in demographics["gender_distribution"].items():
                text += f"    {gender}: {count}\n"

        for metric in ["age", "weight_kg", "height_cm"]:
            if metric in demographics:
                title = metric.replace("_", " ").title()
                text += f"  {title}: "
                text += f"Range {demographics[metric]['min']:.1f}-{demographics[metric]['max']:.1f}, "
                text += f"Mean {demographics[metric]['mean']:.1f}, "
                text += f"Median {demographics[metric]['median']:.1f}\n"

        text += "\n"

        # Test Completion
        test_completion = sensor_summary.get("test_completion", {})
        text += "Test Completion Rates:\n"

        for test_type, data in test_completion.items():
            readable_test = test_type.replace("_", " ").upper()
            text += f"  {readable_test}: {data['completed']} participants "
            text += f"({data['completion_rate']:.1f}%)\n"

        text += "\n"

        # Key Metrics
        metrics = sensor_summary.get("metrics", {})
        if metrics:
            text += "Key Test Metrics (Mean Values):\n"

            for metric, data in metrics.items():
                readable_metric = metric.replace("_", " ").title()
                text += f"  {readable_metric}: {data['mean']:.3f}\n"

        text += "\n"

        # Food Rating Summary
        text += "FOOD RATING ANALYSIS\n"
        text += "-" * 80 + "\n"

        food_summary = report.get("food_rating_summary", {})

        text += f"Participants analyzed: {food_summary.get('participants_analyzed', 0)}\n\n"

        # Rating Statistics
        rating_stats = food_summary.get("rating_statistics", {})

        if rating_stats:
            text += "Rating Statistics:\n"

            for rating_type, data in rating_stats.items():
                readable_type = rating_type.replace("_", " ").title()
                text += f"  {readable_type}: "
                text += f"Mean {data['mean']:.2f}, "
                text += f"Median {data['median']:.2f}, "
                text += f"Range {data['min']:.1f}-{data['max']:.1f}\n"

        text += "\n"

        # Chosen vs Rejected
        chosen_rejected = food_summary.get("chosen_vs_rejected", {})

        if chosen_rejected:
            text += "Rating Differences by Pair Type:\n"

            for pair_type, data in chosen_rejected.items():
                readable_type = pair_type.replace("_", " ").title()
                text += f"  {readable_type}: "
                text += f"Mean {data['mean']:.2f}, "
                text += f"Median {data['median']:.2f}, "
                text += f"Range {data['min']:.1f}-{data['max']:.1f}\n"

        return text

    # The following methods are from the original ParticipantDataAnalyzer class

    def parse_participant_info(self, folder_name):
        """
        Parse participant information from folder name
        Format: Name-Gender-Age-Weight-Height
        Example: Meryemnur-K-21-60-163

        Parameters:
        -----------
        folder_name : str
            Name of the participant's folder

        Returns:
        --------
        dict : Participant information dict with keys: name, gender, age, weight, height
        """
        try:
            # Parse using regex
            pattern = r"([^-]+)-([EK])-(\d+)-(\d+)-(\d+)"
            match = re.match(pattern, folder_name)

            if match:
                name, gender, age, weight, height = match.groups()
                gender_full = "Female" if gender == "K" else "Male"

                return {
                    "name": name,
                    "gender": gender_full,
                    "gender_code": gender,
                    "age": int(age),
                    "weight": int(weight),
                    "height": int(height),
                    "bmi": round(int(weight) / ((int(height) / 100) ** 2), 2)
                }
            else:
                # Try alternative parsing
                parts = folder_name.split('-')
                if len(parts) >= 5:
                    name = parts[0]
                    gender = parts[1]
                    gender_full = "Female" if gender == "K" else "Male"

                    try:
                        age = int(parts[2])
                        weight = int(parts[3])
                        height = int(parts[4].split('.')[0])  # Remove file extension if present

                        return {
                            "name": name,
                            "gender": gender_full,
                            "gender_code": gender,
                            "age": age,
                            "weight": weight,
                            "height": height,
                            "bmi": round(weight / ((height / 100) ** 2), 2)
                        }
                    except (ValueError, IndexError):
                        logger.warning(f"Error parsing participant info from {folder_name}")
                        return None
                else:
                    logger.warning(f"Could not parse participant info from {folder_name}")
                    return None

        except Exception as e:
            logger.error(f"Error parsing participant info: {e}")
            return None

    def list_data_files(self, folder_path):
        """
        List all data files in a folder

        Parameters:
        -----------
        folder_path : str
            Path to the folder

        Returns:
        --------
        list : List of file names
        """
        try:
            files = [f for f in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, f)) and not f.endswith('.py')]

            logger.info(f"Found {len(files)} files in {folder_path}")
            return files
        except Exception as e:
            logger.error(f"Error listing files in {folder_path}: {e}")
            return []

    def categorize_test_files(self, file_list, folder_path):
        """
        Categorize files by test type (pre/post ITUG/ISWAY)

        Parameters:
        -----------
        file_list : list
            List of file names
        folder_path : str
            Path to the folder

        Returns:
        --------
        dict : Dictionary with test types as keys and file paths as values
        """
        test_files = {
            "pre_itug": None,
            "pre_isway": None,
            "post_itug": None,
            "post_isway": None
        }

        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)

            # Determine test type from file name
            file_lower = file_name.lower()

            if "pre" in file_lower and "itug" in file_lower:
                test_files["pre_itug"] = file_path
            elif "pre" in file_lower and "isway" in file_lower:
                test_files["pre_isway"] = file_path
            elif "post" in file_lower and "itug" in file_lower:
                test_files["post_itug"] = file_path
            elif "post" in file_lower and "isway" in file_lower:
                test_files["post_isway"] = file_path
            # If not explicitly labeled, try to infer based on file order
            elif "itug" in file_lower and not test_files["pre_itug"]:
                test_files["pre_itug"] = file_path
            elif "itug" in file_lower and test_files["pre_itug"]:
                test_files["post_itug"] = file_path
            elif "isway" in file_lower and not test_files["pre_isway"]:
                test_files["pre_isway"] = file_path
            elif "isway" in file_lower and test_files["pre_isway"]:
                test_files["post_isway"] = file_path

        # If any file wasn't categorized but contains part_1, part_2, etc., use that
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            file_lower = file_name.lower()

            if "part_1" in file_lower and not test_files["pre_itug"]:
                test_files["pre_itug"] = file_path
            elif "part_2" in file_lower and not test_files["pre_isway"]:
                test_files["pre_isway"] = file_path
            elif "part_3" in file_lower and not test_files["post_itug"]:
                test_files["post_itug"] = file_path
            elif "part_4" in file_lower and not test_files["post_isway"]:
                test_files["post_isway"] = file_path

        # Print summary of categorized files
        for test_type, file_path in test_files.items():
            if file_path:
                logger.info(f"  {test_type}: {os.path.basename(file_path)}")
            else:
                logger.info(f"  {test_type}: Not found")

        return test_files

    def add_to_summary_data(self, participant_summary):
        """
        Add participant data to the summary data list

        Parameters:
        -----------
        participant_summary : dict
            Participant summary data
        """
        # Create a flattened version of the participant summary for the CSV
        flat_summary = {
            "name": participant_summary.get("name", ""),
            "gender": participant_summary.get("gender", ""),
            "age": participant_summary.get("age", ""),
            "weight": participant_summary.get("weight", ""),
            "height": participant_summary.get("height", ""),
            "bmi": participant_summary.get("bmi", ""),
            "initial_rating": participant_summary.get("initial_rating", ""),
            "final_rating": participant_summary.get("final_rating", ""),
            "rejected_pair_rating_diff": participant_summary.get("rejected_pair_rating_diff", ""),
            "chosen_pair_rating_diff": participant_summary.get("chosen_pair_rating_diff", ""),
            "computer_pair_rating_diff": participant_summary.get("computer_pair_rating_diff", ""),
            "timeout_count": participant_summary.get("timeout_count", "")
        }

        # Add test metrics for each test type
        test_types = ["pre_itug", "pre_isway", "post_itug", "post_isway"]

        for test_type in test_types:
            metrics = participant_summary.get(f"{test_type}_metrics", {})

            if metrics:
                # Add key metrics for this test type
                flat_summary[f"{test_type}_duration"] = metrics.get("duration_seconds", "")
                flat_summary[f"{test_type}_num_samples"] = metrics.get("num_samples", "")

                # Add test-specific key metrics
                if "ITUG" in metrics.get("test_type", ""):
                    flat_summary[f"{test_type}_stride_frequency"] = metrics.get("stride_frequency", "")
                elif "ISWAY" in metrics.get("test_type", ""):
                    flat_summary[f"{test_type}_sway_area"] = metrics.get("sway_area", "")

                # Add acceleration metrics for PC1 (first principal component)
                if "PC1_mean" in metrics:
                    flat_summary[f"{test_type}_PC1_mean"] = metrics.get("PC1_mean", "")
                    flat_summary[f"{test_type}_PC1_std"] = metrics.get("PC1_std", "")
                    flat_summary[f"{test_type}_PC1_range"] = metrics.get("PC1_range", "")
                    flat_summary[f"{test_type}_PC1_dominant_freq"] = metrics.get("PC1_dominant_freq", "")
            else:
                # Test data not available
                flat_summary[f"{test_type}_available"] = "No"

        # Add to summary data list
        self.summary_data.append(flat_summary)

    def save_summary_data(self):
        """
        Save the summary data to a CSV file
        """
        if not self.summary_data:
            logger.warning("No summary data to save")
            return

        try:
            # Convert to DataFrame
            summary_df = pd.DataFrame(self.summary_data)

            # Save to CSV
            csv_file = os.path.join(self.output_dir, "summary_data.csv")
            summary_df.to_csv(csv_file, index=False)

            logger.info(f"Saved summary data for {len(summary_df)} participants to {csv_file}")

            # Also save as Excel
            excel_file = os.path.join(self.output_dir, "summary_data.xlsx")
            summary_df.to_excel(excel_file, index=False)

            logger.info(f"Saved summary data as Excel to {excel_file}")

        except Exception as e:
            logger.error(f"Error saving summary data: {e}")
            logger.error(traceback.format_exc())

    def process_test_file(self, file_path, output_dir):
        """
        Process a single test file (ITUG or ISWAY)

        Parameters:
        -----------
        file_path : str
            Path to the test file
        output_dir : str
            Directory to save output files

        Returns:
        --------
        dict : Dictionary with test metrics
        """
        try:
            # Determine test type from file path
            file_name = os.path.basename(file_path)
            file_lower = file_name.lower()

            if "itug" in file_lower:
                test_type = "ITUG"
            elif "isway" in file_lower:
                test_type = "ISWAY"
            else:
                test_type = "Unknown"

            if "pre" in file_lower:
                test_phase = "Pre"
            elif "post" in file_lower:
                test_phase = "Post"
            else:
                test_phase = "Unknown"

            # Load accelerometer data from the file
            data = self.load_accelerometer_data(file_path)

            if data is None or len(data) == 0:
                logger.warning(f"No valid data in {file_name}")
                return {"test_type": f"{test_phase}_{test_type}", "valid": False}

            # Calculate time series length and duration
            num_samples = len(data)
            duration_seconds = num_samples / self.sampling_freq

            logger.info(f"Loaded {num_samples} samples ({duration_seconds:.2f} seconds)")

            # Apply PCA to reduce dimensions and extract primary components
            pca_results = self.apply_pca(data)

            # Check if PCA was successful
            if pca_results is None or len(pca_results) == 0:
                logger.error(f"PCA failed for {file_name}")
                return {"test_type": f"{test_phase}_{test_type}", "valid": False}

            # Create DataFrame with PCA results
            n_components = min(3, pca_results.shape[1])
            pca_cols = [f'PC{i + 1}' for i in range(n_components)]
            pca_df = pd.DataFrame(pca_results[:, :n_components], columns=pca_cols)

            # Calculate basic statistics for PC1
            pc1_stats = {
                'PC1_mean': pca_df['PC1'].mean(),
                'PC1_std': pca_df['PC1'].std(),
                'PC1_range': pca_df['PC1'].max() - pca_df['PC1'].min(),
            }

            # Calculate frequency domain metrics
            freq_metrics = self.calculate_frequency_metrics(pca_df['PC1'].values)

            # Calculate test-specific metrics
            test_specific_metrics = {}

            if test_type == "ITUG":
                # For ITUG, calculate stride-related metrics
                stride_metrics = self.calculate_stride_metrics(pca_df['PC1'].values)
                test_specific_metrics.update(stride_metrics)
            elif test_type == "ISWAY":
                # For ISWAY, calculate sway-related metrics
                if 'PC2' in pca_df.columns:
                    sway_metrics = self.calculate_sway_metrics(pca_df[['PC1', 'PC2']].values)
                    test_specific_metrics.update(sway_metrics)
                else:
                    logger.warning(f"Not enough components for sway metrics in {file_name}")
                    test_specific_metrics = {'sway_area': None, 'mean_distance': None, 'path_length': None}

            # Generate plots
            self.generate_test_plots(pca_df, test_type, test_phase, output_dir)

            # Combine all metrics
            metrics = {
                'test_type': f"{test_phase}_{test_type}",
                'valid': True,
                'num_samples': num_samples,
                'duration_seconds': duration_seconds,
                **pc1_stats,
                **freq_metrics,
                **test_specific_metrics
            }

            # Save metrics to JSON
            metrics_file = os.path.join(output_dir, 'test_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            return metrics

        except Exception as e:
            logger.error(f"Error processing test file {file_path}: {e}")
            logger.error(traceback.format_exc())
            return {"test_type": "Error", "valid": False, "error": str(e)}

    def load_accelerometer_data(self, file_path):
        """
        Load accelerometer data from a file, properly handling string columns

        Parameters:
        -----------
        file_path : str
            Path to the file

        Returns:
        --------
        numpy.ndarray : Array of shape (n_samples, n_features)
        """
        try:
            # Try loading CSV first
            try:
                # Check if it's a CSV with header
                df = pd.read_csv(file_path)

                # Debug: Print the first few rows to see data format
                logger.debug(f"First few rows of the CSV:\n{df.head(2)}")

                # Filter out non-numeric columns
                numeric_df = df.select_dtypes(include=np.number)

                # Check for standard accelerometer columns - more specifically targeting
                accel_cols = [col for col in numeric_df.columns if any(
                    term in col.lower() for term in ['accel', 'acceleration', 'motion'])]

                if accel_cols:
                    logger.info(f"Using acceleration columns: {accel_cols}")
                    return df[accel_cols].values
                else:
                    # If no recognizable columns, use all numeric columns
                    logger.info(
                        f"No specific acceleration columns found. Using all numeric columns: {numeric_df.columns.tolist()}")
                    return numeric_df.values

            except Exception as csv_error:
                # Try as plain text file with space or tab delimiters
                try:
                    data = np.loadtxt(file_path)
                    logger.info(f"Loaded text file with shape: {data.shape}")
                    return data
                except Exception as txt_error:
                    # Try as JSON
                    try:
                        with open(file_path, 'r') as f:
                            data_dict = json.load(f)

                        # Extract accelerometer data from JSON
                        # This depends on the structure of your JSON files
                        if isinstance(data_dict, list):
                            # If it's a list of records
                            accel_data = []
                            for record in data_dict:
                                if isinstance(record, dict) and all(k in record for k in ['x', 'y', 'z']):
                                    accel_data.append([record['x'], record['y'], record['z']])

                            logger.info(f"Loaded JSON list with {len(accel_data)} records")
                            return np.array(accel_data)

                        elif isinstance(data_dict, dict):
                            # If it's a dict with arrays
                            if all(k in data_dict for k in ['x', 'y', 'z']):
                                x = np.array(data_dict['x'])
                                y = np.array(data_dict['y'])
                                z = np.array(data_dict['z'])
                                accel_data = np.column_stack((x, y, z))

                                logger.info(f"Loaded JSON dict with shape: {accel_data.shape}")
                                return accel_data

                        logger.warning(f"Could not extract accelerometer data from JSON")
                        return None

                    except Exception as json_error:
                        logger.error(f"Failed to load file as CSV, text, or JSON")
                        logger.error(f"CSV error: {csv_error}")
                        logger.error(f"Text error: {txt_error}")
                        logger.error(f"JSON error: {json_error}")
                        return None

        except Exception as e:
            logger.error(f"Error loading accelerometer data: {e}")
            return None

    def apply_pca(self, data):
        """
        Apply PCA to accelerometer data following the covariance matrix approach
        described in the paper "Balance Assessment Using a Smartwatch Inertial
        Measurement Unit with Principal Component Analysis for Anatomical Calibration"

        Parameters:
        -----------
        data : numpy.ndarray
            Accelerometer data array of shape (n_samples, n_features)

        Returns:
        --------
        numpy.ndarray : PCA transformed data
        """
        try:
            # Make sure data is numeric
            if not isinstance(data, np.ndarray):
                logger.warning(f"Input data is not a numpy array, attempting to convert")
                data = np.array(data, dtype=float)

            # Check for non-numeric values
            if not np.issubdtype(data.dtype, np.number):
                logger.warning(f"Data contains non-numeric values, forcing conversion to float")
                data = data.astype(float)

            # Check for NaN or infinite values
            if np.isnan(data).any() or np.isinf(data).any():
                logger.warning(f"Data contains NaN or infinite values, replacing with zeros")
                data = np.nan_to_num(data)

            # Basic check for data dimensions
            if len(data.shape) != 2:
                logger.warning(f"Expected 2D array but got {len(data.shape)}D, reshaping")
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                else:
                    data = data.reshape(data.shape[0], -1)

            # Center the data (subtract mean)
            data_centered = data - np.mean(data, axis=0)

            # Calculate covariance matrix manually as described in the paper
            n_samples = data_centered.shape[0]
            cov_matrix = np.zeros((data.shape[1], data.shape[1]))

            for i in range(data.shape[1]):
                for j in range(data.shape[1]):
                    # Calculate covariance between dimensions i and j
                    cov_matrix[i, j] = np.sum(data_centered[:, i] * data_centered[:, j]) / (n_samples - 1)

            # Find eigenvalues and eigenvectors of covariance matrix
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                # Sort eigenvalues and eigenvectors in descending order
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Get number of components to keep
                n_components = min(3, data.shape[1])

                # Extract principal eigenvectors
                principal_eigenvectors = eigenvectors[:, :n_components]

                # Project data onto principal components
                pca_result = np.dot(data_centered, principal_eigenvectors)

                # Log eigenvalues as explained variance
                total_var = eigenvalues.sum()
                explained_variance_ratio = eigenvalues[:n_components] / total_var
                logger.info(f"Applied PCA, explained variance: {explained_variance_ratio}")

                return pca_result

            except np.linalg.LinAlgError:
                logger.error(f"Error computing eigenvalues/eigenvectors, falling back to original implementation")

                # Standardize the data
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)

                # Apply PCA using sklearn (fallback)
                n_components = min(3, data.shape[1])
                if n_components < 1:
                    logger.error(f"Not enough dimensions for PCA ({data.shape[1]})")
                    return data

                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(data_scaled)
                logger.info(f"Applied PCA (fallback), explained variance: {pca.explained_variance_ratio_}")

                return pca_result

        except Exception as e:
            logger.error(f"Error applying PCA: {e}")
            logger.error(traceback.format_exc())

            # If PCA fails, return original data with at most 3 columns
            logger.warning("PCA failed, returning original data (up to 3 columns)")
            if data.shape[1] > 3:
                return data[:, :3]  # Return first 3 columns
            else:
                return data

    def calculate_frequency_metrics(self, signal_data):
        """
        Calculate frequency domain metrics from a signal

        Parameters:
        -----------
        signal_data : numpy.ndarray
            1D signal array

        Returns:
        --------
        dict : Dictionary with frequency metrics
        """
        try:
            # Calculate power spectral density
            f, Pxx = signal.welch(signal_data, fs=self.sampling_freq, nperseg=min(256, len(signal_data)))

            # Find dominant frequency (frequency with maximum power)
            dominant_freq_idx = np.argmax(Pxx)
            dominant_freq = f[dominant_freq_idx]

            # Calculate frequency metrics
            metrics = {
                'PC1_dominant_freq': dominant_freq,
                'PC1_spectral_entropy': -np.sum(Pxx * np.log2(Pxx + 1e-10)) / np.log2(len(Pxx)),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating frequency metrics: {e}")
            return {'PC1_dominant_freq': None, 'PC1_spectral_entropy': None}

    def calculate_stride_metrics(self, signal_data):
        """
        Calculate stride-related metrics for ITUG test

        Parameters:
        -----------
        signal_data : numpy.ndarray
            1D signal array

        Returns:
        --------
        dict : Dictionary with stride metrics
        """
        try:
            # Detect peaks in the signal
            # These peaks can correspond to steps in the gait cycle
            peaks, _ = signal.find_peaks(signal_data, distance=0.5 * self.sampling_freq)

            if len(peaks) < 2:
                # Not enough peaks to calculate stride metrics
                return {'stride_frequency': None, 'num_strides': 0}

            # Calculate average time between peaks (stride time)
            stride_times = np.diff(peaks) / self.sampling_freq
            avg_stride_time = np.mean(stride_times)

            # Calculate stride frequency (strides per second)
            stride_frequency = 1 / avg_stride_time if avg_stride_time > 0 else None

            return {
                'stride_frequency': stride_frequency,
                'num_strides': len(peaks) - 1  # Number of complete strides
            }

        except Exception as e:
            logger.error(f"Error calculating stride metrics: {e}")
            return {'stride_frequency': None, 'num_strides': 0}

    def calculate_sway_metrics(self, sway_data):
        """
        Calculate sway-related metrics for ISWAY test

        Parameters:
        -----------
        sway_data : numpy.ndarray
            2D array with sway data (x and y coordinates)

        Returns:
        --------
        dict : Dictionary with sway metrics
        """
        try:
            # Extract x and y coordinates
            x = sway_data[:, 0]
            y = sway_data[:, 1]

            # Calculate basic sway metrics

            # 1. Sway area (area of convex hull)
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(sway_data)
                sway_area = hull.volume  # In 2D, volume is actually area
            except Exception as hull_error:
                # Approximate using standard deviations
                sway_area = np.pi * np.std(x) * np.std(y)
                logger.warning(f"Using approximation for sway area: {hull_error}")

            # 2. Mean distance from center
            center_x, center_y = np.mean(x), np.mean(y)
            distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mean_distance = np.mean(distances)

            # 3. Path length (total distance traveled)
            path_length = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

            return {
                'sway_area': sway_area,
                'mean_distance': mean_distance,
                'path_length': path_length
            }

        except Exception as e:
            logger.error(f"Error calculating sway metrics: {e}")
            return {'sway_area': None, 'mean_distance': None, 'path_length': None}

    def generate_test_plots(self, pca_df, test_type, test_phase, output_dir):
        """
        Generate plots for test data visualization

        Parameters:
        -----------
        pca_df : pd.DataFrame
            DataFrame with PCA results
        test_type : str
            Test type (ITUG or ISWAY)
        test_phase : str
            Test phase (Pre or Post)
        output_dir : str
            Directory to save plots
        """
        try:
            # 1. Time series plot of principal components
            plt.figure(figsize=(12, 6))
            time_axis = np.arange(len(pca_df)) / self.sampling_freq

            for col, color in zip(['PC1', 'PC2', 'PC3'], ['b', 'g', 'r']):
                plt.plot(time_axis, pca_df[col], color=color, label=col)

            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'{test_phase} {test_type} - Principal Components')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'principal_components.png'))

            # 2. Trajectory plot (PC1 vs PC2)
            plt.figure(figsize=(8, 8))
            plt.scatter(pca_df['PC1'], pca_df['PC2'], s=3, alpha=0.5)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(f'{test_phase} {test_type} - PC1 vs PC2 Trajectory')
            plt.grid(alpha=0.3)
            # Add start and end points
            plt.scatter([pca_df['PC1'].iloc[0]], [pca_df['PC2'].iloc[0]], color='g', s=100, label='Start')
            plt.scatter([pca_df['PC1'].iloc[-1]], [pca_df['PC2'].iloc[-1]], color='r', s=100, label='End')
            plt.legend()
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pc_trajectory.png'))

            # 3. Power spectrum of PC1
            plt.figure(figsize=(10, 6))
            f, Pxx = signal.welch(pca_df['PC1'].values, fs=self.sampling_freq, nperseg=min(256, len(pca_df)))
            plt.semilogy(f, Pxx)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density')
            plt.title(f'{test_phase} {test_type} - PC1 Power Spectrum')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'power_spectrum.png'))

            # Close all figures to free memory
            plt.close('all')

        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            logger.error(traceback.format_exc())

    def generate_sensor_aggregate_visualizations(self, output_dir):
        """
        Generate aggregate visualizations from the summary data
        """
        if not self.summary_data:
            logger.warning("No summary data for visualizations")
            return

        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.summary_data)

            # Create visualization directory
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # 1. Age distribution by gender
            plt.figure(figsize=(10, 6))

            if 'gender' in df.columns and 'age' in df.columns:
                # Check if we have both males and females
                genders = df['gender'].unique()

                if len(genders) > 1:
                    # Create separate histograms by gender
                    for gender, color, alpha in zip(['Male', 'Female'], ['blue', 'red'], [0.5, 0.5]):
                        gender_data = df[df['gender'] == gender]['age']
                        if not gender_data.empty:
                            plt.hist(gender_data, bins=range(10, 90, 5), alpha=alpha, color=color, label=gender)
                else:
                    # Single gender, use default histogram
                    plt.hist(df['age'], bins=range(10, 90, 5))

                plt.xlabel('Age (years)')
                plt.ylabel('Number of Participants')
                plt.title('Age Distribution by Gender')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(viz_dir, 'age_distribution.png'))

            # 2. Height vs. Weight scatter plot with BMI coloring
            plt.figure(figsize=(10, 6))

            if all(col in df.columns for col in ['height', 'weight', 'bmi', 'gender']):
                # Create scatter plot
                for gender, marker in zip(['Male', 'Female'], ['o', 's']):
                    gender_data = df[df['gender'] == gender]
                    if not gender_data.empty:
                        scatter = plt.scatter(
                            gender_data['height'],
                            gender_data['weight'],
                            c=gender_data['bmi'],
                            cmap='viridis',
                            alpha=0.7,
                            marker=marker,
                            label=gender
                        )

                plt.colorbar(scatter, label='BMI')
                plt.xlabel('Height (cm)')
                plt.ylabel('Weight (kg)')
                plt.title('Height vs. Weight (Colored by BMI)')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(viz_dir, 'height_weight_bmi.png'))

            # 3. Pre vs. Post test comparisons
            # For ITUG
            self._create_pre_post_comparison(df, 'pre_itug_PC1_mean', 'post_itug_PC1_mean',
                                             'ITUG PC1 Mean (Pre vs. Post)', viz_dir)

            # For ISWAY
            self._create_pre_post_comparison(df, 'pre_isway_sway_area', 'post_isway_sway_area',
                                             'ISWAY Sway Area (Pre vs. Post)', viz_dir)

            # 4. Initial vs. Final rating
            self._create_pre_post_comparison(df, 'initial_rating', 'final_rating',
                                             'Initial vs. Final Rating', viz_dir)

            # 5. Rating differences comparison
            plt.figure(figsize=(10, 6))

            rating_cols = ['rejected_pair_rating_diff', 'chosen_pair_rating_diff', 'computer_pair_rating_diff']

            if all(col in df.columns for col in rating_cols):
                # Get data for each rating type
                data = [df[col].dropna() for col in rating_cols]
                labels = ['Rejected Pair', 'Chosen Pair', 'Computer Pair']

                # Create box plot
                plt.boxplot(data, labels=labels)
                plt.ylabel('Rating Difference')
                plt.title('Rating Differences Comparison')
                plt.grid(axis='y', alpha=0.3)
                plt.savefig(os.path.join(viz_dir, 'rating_differences.png'))

            plt.close('all')
            logger.info(f"Generated aggregate visualizations in {viz_dir}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            logger.error(traceback.format_exc())

    def _create_pre_post_comparison(self, df, pre_col, post_col, title, output_dir):
        """
        Create a pre vs. post comparison plot

        Parameters:
        -----------
        df : pd.DataFrame
            Data frame with pre and post columns
        pre_col : str
            Name of pre-test column
        post_col : str
            Name of post-test column
        title : str
            Plot title
        output_dir : str
            Directory to save the plot
        """
        if pre_col in df.columns and post_col in df.columns:
            # Get participants with both pre and post data
            valid_data = df[[pre_col, post_col]].dropna()

            if len(valid_data) > 0:
                plt.figure(figsize=(10, 6))

                # Create scatter plot
                plt.scatter(valid_data[pre_col], valid_data[post_col], alpha=0.7)

                # Add diagonal line (y=x)
                min_val = min(valid_data[pre_col].min(), valid_data[post_col].min())
                max_val = max(valid_data[pre_col].max(), valid_data[post_col].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

                plt.xlabel(f"Pre ({pre_col})")
                plt.ylabel(f"Post ({post_col})")
                plt.title(title)
                plt.grid(alpha=0.3)

                # Equal aspect ratio
                plt.axis('equal')

                # Save plot
                filename = f"{pre_col}_vs_{post_col}.png"
                plt.savefig(os.path.join(output_dir, filename))
                plt.close()