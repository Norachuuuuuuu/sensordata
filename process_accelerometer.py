import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Main data processing class
class ParticipantDataAnalyzer:
    def __init__(self, root_folder="18-04-2025SENSORDATA", sampling_freq=100):
        """
        Initialize the analyzer with the root folder containing all participant data
        
        Parameters:
        -----------
        root_folder : str
            Path to the folder containing all participant data folders
        sampling_freq : float
            Sampling frequency of the accelerometer data in Hz
        """
        self.root_folder = root_folder
        self.sampling_freq = sampling_freq
        self.output_dir = "analysis_results"
        self.summary_data = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {os.path.abspath(self.output_dir)}")
    
    def analyze_all_participants(self):
        """
        Analyze data from all participants in the root folder
        """
        print(f"\n{'='*50}")
        print(f"Analyzing all participants in {self.root_folder}")
        print(f"{'='*50}")
        
        # Get all participant folders
        try:
            participant_folders = [f for f in os.listdir(self.root_folder) 
                                  if os.path.isdir(os.path.join(self.root_folder, f))]
            
            print(f"Found {len(participant_folders)} participant folders")
            
            # Process each participant
            for i, folder in enumerate(participant_folders):
                print(f"\nProcessing participant {i+1}/{len(participant_folders)}: {folder}")
                self.process_participant(folder)
            
            # Save summary data to CSV
            self.save_summary_data()
            
            # Generate aggregate visualizations
            self.generate_aggregate_visualizations()
            
            print(f"\nCompleted analysis of all {len(participant_folders)} participants")
            print(f"Summary data saved to {os.path.join(self.output_dir, 'summary_data.csv')}")
            
            # Print some aggregate statistics
            self.print_aggregate_statistics()
            
        except Exception as e:
            print(f"Error analyzing participants: {e}")
            import traceback
            traceback.print_exc()
    
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
            print("No summary data to save")
            return
        
        try:
            # Convert to DataFrame
            summary_df = pd.DataFrame(self.summary_data)
            
            # Save to CSV
            csv_file = os.path.join(self.output_dir, "summary_data.csv")
            summary_df.to_csv(csv_file, index=False)
            
            print(f"Saved summary data for {len(summary_df)} participants to {csv_file}")
            
            # Also save as Excel
            excel_file = os.path.join(self.output_dir, "summary_data.xlsx")
            summary_df.to_excel(excel_file, index=False)
            
            print(f"Saved summary data as Excel to {excel_file}")
            
        except Exception as e:
            print(f"Error saving summary data: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_aggregate_visualizations(self):
        """
        Generate aggregate visualizations from the summary data
        """
        if not self.summary_data:
            print("No summary data for visualizations")
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.summary_data)
            
            # Create visualization directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
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
            print(f"Generated aggregate visualizations in {viz_dir}")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    def print_aggregate_statistics(self):
        """
        Print aggregate statistics from the summary data
        """
        if not self.summary_data:
            print("No summary data for statistics")
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.summary_data)
            
            print("\n" + "="*50)
            print("AGGREGATE STATISTICS")
            print("="*50)
            
            # Participant demographics
            print("\nParticipant Demographics:")
            print(f"  Total participants: {len(df)}")
            
            if 'gender' in df.columns:
                gender_counts = df['gender'].value_counts()
                for gender, count in gender_counts.items():
                    print(f"  {gender}: {count} ({count/len(df)*100:.1f}%)")
            
            if 'age' in df.columns:
                age = df['age'].dropna()
                if len(age) > 0:
                    print(f"  Age range: {age.min()}-{age.max()} years (mean: {age.mean():.1f})")
            
            if 'weight' in df.columns:
                weight = df['weight'].dropna()
                if len(weight) > 0:
                    print(f"  Weight range: {weight.min()}-{weight.max()} kg (mean: {weight.mean():.1f})")
            
            if 'height' in df.columns:
                height = df['height'].dropna()
                if len(height) > 0:
                    print(f"  Height range: {height.min()}-{height.max()} cm (mean: {height.mean():.1f})")
            
            if 'bmi' in df.columns:
                bmi = df['bmi'].dropna()
                if len(bmi) > 0:
                    print(f"  BMI range: {bmi.min():.1f}-{bmi.max():.1f} (mean: {bmi.mean():.1f})")
            
            # Test completion rates
            print("\nTest Completion Rates:")
            test_types = ["pre_itug", "pre_isway", "post_itug", "post_isway"]
            
            for test_type in test_types:
                col_name = f"{test_type}_duration"
                if col_name in df.columns:
                    complete_count = df[col_name].notna().sum()
                    print(f"  {test_type.replace('_', ' ').upper()}: {complete_count}/{len(df)} ({complete_count/len(df)*100:.1f}%)")
            
            # Rating statistics
            print("\nRating Statistics:")
            rating_cols = ['initial_rating', 'final_rating']
            
            for col in rating_cols:
                if col in df.columns:
                    ratings = df[col].dropna()
                    if len(ratings) > 0:
                        print(f"  {col.replace('_', ' ').title()}: Mean = {ratings.mean():.2f}, Median = {ratings.median():.2f}, Range = {ratings.min():.1f}-{ratings.max():.1f}")
            
            # Rating difference statistics
            rating_diff_cols = ['rejected_pair_rating_diff', 'chosen_pair_rating_diff', 'computer_pair_rating_diff']
            
            for col in rating_diff_cols:
                if col in df.columns:
                    diff = df[col].dropna()
                    if len(diff) > 0:
                        print(f"  {col.replace('_', ' ').title()}: Mean = {diff.mean():.2f}, Median = {diff.median():.2f}, Range = {diff.min():.1f}-{diff.max():.1f}")
            
            # Timeout counts
            if 'timeout_count' in df.columns:
                timeout = df['timeout_count'].dropna()
                if len(timeout) > 0:
                    print(f"\nTimeout Count: Mean = {timeout.mean():.2f}, Median = {timeout.median():.2f}, Max = {timeout.max():.0f}")
            
            # Test metrics
            print("\nTest Metrics (Mean Values):")
            
            # ITUG metrics
            itug_metrics = [('stride_frequency', 'Stride Frequency (Hz)')]
            
            print("  ITUG Test:")
            for metric_name, display_name in itug_metrics:
                pre_col = f"pre_itug_{metric_name}"
                post_col = f"post_itug_{metric_name}"
                
                if pre_col in df.columns and post_col in df.columns:
                    pre_data = df[pre_col].dropna()
                    post_data = df[post_col].dropna()
                    
                    if len(pre_data) > 0 and len(post_data) > 0:
                        pre_mean = pre_data.mean()
                        post_mean = post_data.mean()
                        
                        print(f"    {display_name}: Pre = {pre_mean:.2f}, Post = {post_mean:.2f}, Change = {(post_mean - pre_mean):.2f} ({(post_mean - pre_mean)/pre_mean*100:.1f}%)")
            
            # ISWAY metrics
            isway_metrics = [('sway_area', 'Sway Area')]
            
            print("  ISWAY Test:")
            for metric_name, display_name in isway_metrics:
                pre_col = f"pre_isway_{metric_name}"
                post_col = f"post_isway_{metric_name}"
                
                if pre_col in df.columns and post_col in df.columns:
                    pre_data = df[pre_col].dropna()
                    post_data = df[post_col].dropna()
                    
                    if len(pre_data) > 0 and len(post_data) > 0:
                        pre_mean = pre_data.mean()
                        post_mean = post_data.mean()
                        
                        print(f"    {display_name}: Pre = {pre_mean:.2f}, Post = {post_mean:.2f}, Change = {(post_mean - pre_mean):.2f} ({(post_mean - pre_mean)/pre_mean*100:.1f}%)")
                        
            # PC1 metrics for all tests
            print("\nPC1 Metrics (Mean Values):")
            for test_type in test_types:
                pc1_mean_col = f"{test_type}_PC1_mean"
                pc1_std_col = f"{test_type}_PC1_std"
                
                if pc1_mean_col in df.columns and pc1_std_col in df.columns:
                    pc1_mean = df[pc1_mean_col].dropna()
                    pc1_std = df[pc1_std_col].dropna()
                    
                    if len(pc1_mean) > 0 and len(pc1_std) > 0:
                        print(f"  {test_type.replace('_', ' ').upper()}: Mean = {pc1_mean.mean():.3f}, StdDev = {pc1_std.mean():.3f}")
            
            # Save statistics to file
            stats_file = os.path.join(self.output_dir, "aggregate_statistics.txt")
            with open(stats_file, 'w') as f:
                f.write("AGGREGATE STATISTICS\n")
                f.write("="*50 + "\n\n")
                
                f.write("Participant Demographics:\n")
                f.write(f"  Total participants: {len(df)}\n")
                
                if 'gender' in df.columns:
                    gender_counts = df['gender'].value_counts()
                    for gender, count in gender_counts.items():
                        f.write(f"  {gender}: {count} ({count/len(df)*100:.1f}%)\n")
                
                if 'age' in df.columns:
                    age = df['age'].dropna()
                    if len(age) > 0:
                        f.write(f"  Age range: {age.min()}-{age.max()} years (mean: {age.mean():.1f})\n")
                
                # Write remaining statistics to file...
                # (Similar to the print statements above)
            
            print(f"\nAggregate statistics saved to {stats_file}")
            
        except Exception as e:
            print(f"Error calculating aggregate statistics: {e}")
            import traceback
            traceback.print_exc()
    
    def process_participant(self, participant_folder):
        """
        Process data for a single participant
        
        Parameters:
        -----------
        participant_folder : str
            Name of the participant's folder
        """
        try:
            # Parse participant info from folder name
            participant_info = self.parse_participant_info(participant_folder)
            if not participant_info:
                print(f"Could not parse participant info from folder name: {participant_folder}")
                return
            
            print(f"Processing data for {participant_info['name']}, {participant_info['gender']}, "
                  f"{participant_info['age']} years, {participant_info['weight']} kg, "
                  f"{participant_info['height']} cm")
            
            # Create participant output directory
            participant_output_dir = os.path.join(self.output_dir, participant_folder)
            os.makedirs(participant_output_dir, exist_ok=True)
            
            # Get all files in participant folder
            folder_path = os.path.join(self.root_folder, participant_folder)
            all_files = self.list_data_files(folder_path)
            
            # Categorize files by test type
            test_files = self.categorize_test_files(all_files, folder_path)
            
            # Process each test type
            test_results = {}
            for test_type, file_path in test_files.items():
                if file_path:
                    print(f"\nProcessing {test_type} test")
                    test_output_dir = os.path.join(participant_output_dir, test_type)
                    os.makedirs(test_output_dir, exist_ok=True)
                    
                    # Process the file for this test type
                    result = self.process_test_file(file_path, test_output_dir)
                    test_results[test_type] = result
            
            # Look for ratings data file
            ratings_data = self.load_ratings_data(folder_path)
            
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
            
            print(f"Completed processing for {participant_info['name']}")
            
        except Exception as e:
            print(f"Error processing participant {participant_folder}: {e}")
            import traceback
            traceback.print_exc()
    
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
                    "bmi": round(int(weight) / ((int(height)/100)**2), 2)
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
                        height = int(parts[4])
                        
                        return {
                            "name": name,
                            "gender": gender_full,
                            "gender_code": gender,
                            "age": age,
                            "weight": weight,
                            "height": height,
                            "bmi": round(weight / ((height/100)**2), 2)
                        }
                    except (ValueError, IndexError):
                        print(f"Error parsing participant info from {folder_name}")
                        return None
                else:
                    print(f"Could not parse participant info from {folder_name}")
                    return None
                    
        except Exception as e:
            print(f"Error parsing participant info: {e}")
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
            
            print(f"Found {len(files)} files in {folder_path}")
            return files
        except Exception as e:
            print(f"Error listing files in {folder_path}: {e}")
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
                print(f"  {test_type}: {os.path.basename(file_path)}")
            else:
                print(f"  {test_type}: Not found")
        
        return test_files
    
    def load_ratings_data(self, folder_path):
        """
        Look for and load ratings data file in the participant's folder
        
        Parameters:
        -----------
        folder_path : str
            Path to the participant's folder
            
        Returns:
        --------
        dict : Dictionary with ratings data
        """
        # Initialize ratingsBased on your request, it seems the code needs to be completed to handle the rating data and finish the `ParticipantDataAnalyzer` class. The main missing parts appear to be the methods for processing test files and loading ratings data. Here are the missing parts to add to your initial code:

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
            print(f"  No valid data in {file_name}")
            return {"test_type": f"{test_phase}_{test_type}", "valid": False}
        
        # Calculate time series length and duration
        num_samples = len(data)
        duration_seconds = num_samples / self.sampling_freq
        
        print(f"  Loaded {num_samples} samples ({duration_seconds:.2f} seconds)")
        
        # Apply PCA to reduce dimensions and extract primary components
        pca_results = self.apply_pca(data)
        pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2', 'PC3'])
        
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
            sway_metrics = self.calculate_sway_metrics(pca_df[['PC1', 'PC2']].values)
            test_specific_metrics.update(sway_metrics)
        
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
        print(f"  Error processing test file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"test_type": "Error", "valid": False, "error": str(e)}

def load_accelerometer_data(self, file_path):
    """
    Load accelerometer data from a file
    
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
            
            # Check for standard accelerometer columns
            accel_cols = [col for col in df.columns if any(
                term in col.lower() for term in ['acc', 'accel', 'accelerometer', 'x', 'y', 'z'])]
            
            if accel_cols:
                print(f"  Loaded CSV with columns: {accel_cols}")
                return df[accel_cols].values
            else:
                # If no recognizable columns, use all numeric columns
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                print(f"  Loaded CSV with numeric columns: {numeric_cols}")
                return df[numeric_cols].values
                
        except Exception as csv_error:
            # Try as plain text file with space or tab delimiters
            try:
                data = np.loadtxt(file_path)
                print(f"  Loaded text file with shape: {data.shape}")
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
                        
                        print(f"  Loaded JSON list with {len(accel_data)} records")
                        return np.array(accel_data)
                    
                    elif isinstance(data_dict, dict):
                        # If it's a dict with arrays
                        if all(k in data_dict for k in ['x', 'y', 'z']):
                            x = np.array(data_dict['x'])
                            y = np.array(data_dict['y'])
                            z = np.array(data_dict['z'])
                            accel_data = np.column_stack((x, y, z))
                            
                            print(f"  Loaded JSON dict with shape: {accel_data.shape}")
                            return accel_data
                    
                    print(f"  Could not extract accelerometer data from JSON")
                    return None
                    
                except Exception as json_error:
                    print(f"  Failed to load file as CSV, text, or JSON")
                    print(f"  CSV error: {csv_error}")
                    print(f"  Text error: {txt_error}")
                    print(f"  JSON error: {json_error}")
                    return None
    
    except Exception as e:
        print(f"  Error loading accelerometer data: {e}")
        return None

def apply_pca(self, data):
    """
    Apply PCA to accelerometer data
    
    Parameters:
    -----------
    data : numpy.ndarray
        Accelerometer data array of shape (n_samples, n_features)
        
    Returns:
    --------
    numpy.ndarray : PCA transformed data
    """
    try:
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Apply PCA
        pca = PCA(n_components=min(3, data.shape[1]))
        pca_result = pca.fit_transform(data_scaled)
        
        print(f"  Applied PCA, explained variance: {pca.explained_variance_ratio_}")
        
        return pca_result
        
    except Exception as e:
        print(f"  Error applying PCA: {e}")
        # Return the original data if PCA fails
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
        print(f"  Error calculating frequency metrics: {e}")
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
        peaks, _ = signal.find_peaks(signal_data, distance=0.5*self.sampling_freq)
        
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
        print(f"  Error calculating stride metrics: {e}")
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
        except:
            # Approximate using standard deviations
            sway_area = np.pi * np.std(x) * np.std(y)
        
        # 2. Mean distance from center
        center_x, center_y = np.mean(x), np.mean(y)
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mean_distance = np.mean(distances)
        
        # 3. Path length (total distance traveled)
        path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        
        return {
            'sway_area': sway_area,
            'mean_distance': mean_distance,
            'path_length': path_length
        }
        
    except Exception as e:
        print(f"  Error calculating sway metrics: {e}")
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
        print(f"  Error generating plots: {e}")
        import traceback
        traceback.print_exc()

def load_ratings_data(self, folder_path):
    """
    Look for and load ratings data file in the participant's folder
    
    Parameters:
    -----------
    folder_path : str
        Path to the participant's folder
        
    Returns:
    --------
    dict : Dictionary with ratings data
    """
    ratings_data = {
        "initial_rating": None,
        "final_rating": None,
        "rejected_pair_rating_diff": None,
        "chosen_pair_rating_diff": None,
        "computer_pair_rating_diff": None,
        "timeout_count": None
    }
    
    # First check if foodratingexperiment subfolder exists
    food_rating_folder = os.path.join(folder_path, "foodratingexperiment")
    if os.path.exists(food_rating_folder) and os.path.isdir(food_rating_folder):
        print(f"  Found foodratingexperiment subfolder")
        search_path = food_rating_folder
    else:
        # Fall back to the original folder if foodratingexperiment subfolder doesn't exist
        print(f"  foodratingexperiment subfolder not found, checking main folder")
        search_path = folder_path
    try:
        # Search for ratings files
        ratings_file_patterns = [
            'ratings.json', 'ratings.csv', 'ratings.txt',
            'rating.json', 'rating.csv', 'rating.txt',
            'survey.json', 'survey.csv', 'survey.txt'
        ]
        
        ratings_file = None
        for pattern in ratings_file_patterns:
            for file in os.listdir(search_path):
                if pattern.lower() in file.lower():
                    ratings_file = os.path.join(search_path, file)
                    break
            if ratings_file:
                break

        if not ratings_file:
            print("  No ratings file found")
            return ratings_data

        print(f"  Found ratings file: {os.path.basename(ratings_file)}")

        # Load ratings data based on file type
        if ratings_file.endswith('.json'):
            # Load JSON
            with open(ratings_file, 'r') as f:
                data = json.load(f)
            
            # Extract ratings
            if isinstance(data, dict):
                # Direct mapping if keys match
                for key in ratings_data.keys():
                    if key in data:
                        ratings_data[key] = data[key]
                
                # Try alternative keys
                key_mappings = {
                    "initial_rating": ["initial", "pre", "pre_rating", "preRating", "initialRating"],
                    "final_rating": ["final", "post", "post_rating", "postRating", "finalRating"],
                    "rejected_pair_rating_diff": ["rejected", "rejected_diff", "rejectedPair", "rejectedDiff"],
                    "chosen_pair_rating_diff": ["chosen", "chosen_diff", "chosenPair", "chosenDiff"],
                    "computer_pair_rating_diff": ["computer", "computer_diff", "computerPair", "computerDiff"],
                    "timeout_count": ["timeout", "timeouts", "timeoutCount"]
                }
                
                for target_key, possible_keys in key_mappings.items():
                    if ratings_data[target_key] is None:
                        for possible_key in possible_keys:
                                if possible_key in data:
                                    ratings_data[target_key] = data[possible_key]
                                break
        elif ratings_file.endswith('.csv'):
            # Load CSV
            df = pd.read_csv(ratings_file)
        
        # Try to find columns for each rating
        column_mappings = {
            "initial_rating": ["initial", "pre", "pre_rating", "preRating", "initialRating"],
            "final_rating": ["final", "post", "post_rating", "postRating", "finalRating"],
            "rejected_pair_rating_diff": ["rejected", "rejected_diff", "rejectedPair", "rejectedDiff"],
            "chosen_pair_rating_diff": ["chosen", "chosen_diff", "chosenPair", "chosenDiff"],
            "computer_pair_rating_diff": ["computer", "computer_diff", "computerPair", "computerDiff"],
            "timeout_count": ["timeout", "timeouts", "timeoutCount"]
        }
        
        for target_key, possible_columns in column_mappings.items():
            for col_name in df.columns:
                if any(possible_col.lower() in col_name.lower() for possible_col in possible_columns):
                    # Take first row value
                    ratings_data[target_key] = df[col_name].iloc[0]
                    break
            elif ratings_file.endswith('.txt'):
        # Load text file
        with open(ratings_file, 'r') as f:
            lines = f.readlines()
        
        # Try to parse each line for rating data
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                try:
                    value = float(value)  # Convert to number if possible
                except:
                    continue  # Skip if not a number
                
                # Check each rating type
                if any(term in key for term in ["initial", "pre"]):
                    ratings_data["initial_rating"] = value
                elif any(term in key for term in ["final", "post"]):
                    ratings_data["final_rating"] = value
                elif any(term in key for term in ["rejected", "reject"]):
                    ratings_data["rejected_pair_rating_diff"] = value
                elif any(term in key for term in ["chosen", "choose"]):
                    ratings_data["chosen_pair_rating_diff"] = value
                elif any(term in key for term in ["computer", "auto"]):
                    ratings_data["computer_pair_rating_diff"] = value
                elif any(term in key for term in ["timeout", "time out"]):
                    ratings_data["timeout_count"] = value

    print(f"  Loaded ratings data: {ratings_data}")
    return ratings_data

except Exception as e:
    print(f"  Error loading ratings data: {e}")
    return ratings_data