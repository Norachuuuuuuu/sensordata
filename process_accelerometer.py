import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import re
import json

# Constants for test types
PRE_ITUG = 'pre_itug'
POST_ITUG = 'post_itug'
PRE_ISWAY = 'pre_isway'
POST_ISWAY = 'post_isway'

# Function to parse participant information from filename
def parse_participant_info(filename):
    # Extract the participant information using regex
    # Format: name-gender-age-weight-height
    pattern = r'([A-Za-z]+)-([EK])-(\d+)-(\d+)-(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        name = match.group(1)
        gender = 'Male' if match.group(2) == 'E' else 'Female'
        age = int(match.group(3))
        weight = int(match.group(4))
        height = int(match.group(5))
        
        return {
            'name': name,
            'gender': gender,
            'age': age,
            'weight': weight,
            'height': height,
            'filename': filename
        }
    else:
        print(f"Warning: Could not parse participant info from filename: {filename}")
        return None

# Function to identify test type (PRE_ITUG, POST_ITUG, PRE_ISWAY, POST_ISWAY)
def identify_test_type(filename):
    filename_lower = filename.lower()
    
    if 'pre' in filename_lower and 'itug' in filename_lower:
        return PRE_ITUG
    elif 'post' in filename_lower and 'itug' in filename_lower:
        return POST_ITUG
    elif 'pre' in filename_lower and 'isway' in filename_lower:
        return PRE_ISWAY
    elif 'post' in filename_lower and 'isway' in filename_lower:
        return POST_ISWAY
    else:
        # Default to None if unable to identify
        print(f"Warning: Unable to identify test type for file: {filename}")
        return None

# Function to list all participant folders
def list_participant_folders(base_dir):
    print(f"Looking for participant folders in: {base_dir}")
    
    # Check if the base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' does not exist!")
        return []
    
    # Get all subfolders in the base directory
    participant_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    print(f"Found {len(participant_folders)} participant folders")
    for i, folder in enumerate(participant_folders):
        print(f"{i+1}. {folder}")
    
    return participant_folders

# Function to list data files for a participant
def list_participant_data_files(participant_folder_path):
    print(f"\nExamining participant folder: {participant_folder_path}")
    
    files = [f for f in os.listdir(participant_folder_path) if os.path.isfile(os.path.join(participant_folder_path, f))]
    data_files = [f for f in files if not f.endswith('.py') and not f.startswith('.')]
    
    print(f"Available files for this participant: {len(data_files)}")
    for i, file in enumerate(data_files):
        print(f"  {i+1}. {file}")
    
    return data_files

# Load your accelerometer data
def load_data(file_path):
    print(f"Attempting to load data from {file_path}")
    
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        # Try different loading methods based on extension
        if ext in ['.csv', '.txt']:
            # Try different delimiters for CSV-like files
            for delimiter in [',', ';', '\t', ' ']:
                try:
                    data = pd.read_csv(file_path, delimiter=delimiter)
                    if len(data.columns) > 1:  # Successful parse should have multiple columns
                        print(f"Successfully loaded data with delimiter '{delimiter}'")
                        print(f"Data shape: {data.shape}")
                        print(f"Columns: {data.columns.tolist()}")
                        return data
                except Exception as e:
                    continue
            
            # If none worked, try with default settings
            data = pd.read_csv(file_path, on_bad_lines='skip')
            print(f"Loaded data with some parsing errors. Shape: {data.shape}")
            return data
            
        elif ext in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
            print(f"Successfully loaded Excel data. Shape: {data.shape}")
            return data
            
        elif ext in ['.json']:
            data = pd.read_json(file_path)
            print(f"Successfully loaded JSON data. Shape: {data.shape}")
            return data
            
        else:
            # Try to infer the format
            try:
                data = pd.read_csv(file_path)
                print(f"Inferred CSV format. Shape: {data.shape}")
                return data
            except:
                try:
                    data = pd.read_excel(file_path)
                    print(f"Inferred Excel format. Shape: {data.shape}")
                    return data
                except Exception as e:
                    print(f"Could not load file. Error: {e}")
                    raise
    
    except Exception as main_error:
        print(f"Error loading data: {main_error}")
        raise

# Preprocess the data to handle missing values
def preprocess_data(data):
    print("Preprocessing data and handling missing values")
    
    # Display info about missing values
    missing_values = data.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    
    # Identify numeric and time columns
    time_columns = [col for col in data.columns if col.lower() in ['time', 'timestamp', 'datetime']]
    numeric_columns = data.select_dtypes(include=[np.number]).columns.difference(time_columns)
    
    # Create a copy of the data
    processed_data = data.copy()
    
    # Handle missing values in numeric columns
    if len(numeric_columns) > 0:
        # Use a simple imputer to fill missing values with the mean
        imputer = SimpleImputer(strategy='mean')
        processed_data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        
        print(f"Imputed missing values in {len(numeric_columns)} numeric columns")
    
    # Handle missing values in non-numeric columns
    non_numeric_cols = data.columns.difference(numeric_columns)
    for col in non_numeric_cols:
        if data[col].isnull().any():
            if pd.api.types.is_datetime64_dtype(data[col]):
                # For datetime columns, forward fill then backward fill
                processed_data[col] = processed_data[col].fillna(method='ffill').fillna(method='bfill')
            else:
                # For other non-numeric columns, fill with the most frequent value
                most_frequent = data[col].mode()[0]
                processed_data[col] = processed_data[col].fillna(most_frequent)
    
    # Verify no missing values remain
    remaining_nulls = processed_data.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"Warning: {remaining_nulls} missing values remain after imputation")
        # Last resort - drop rows with any remaining missing values
        processed_data = processed_data.dropna()
        print(f"Dropped rows with missing values. New shape: {processed_data.shape}")
    else:
        print("Successfully handled all missing values")
    
    return processed_data

# Apply Butterworth filter
def apply_butterworth_filter(data, sampling_freq, cutoff_freq=3.5):
    print(f"Applying {cutoff_freq} Hz cutoff, zero-phase, 4th order Butterworth filter")
    
    # Design the Butterworth filter
    nyquist_freq = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    
    # Apply zero-phase filtering (forward and backward)
    filtered_data = data.copy()
    
    # Identify numeric columns (likely accelerometer data)
    time_columns = [col for col in data.columns if col.lower() in ['time', 'timestamp', 'datetime']]
    numeric_columns = data.select_dtypes(include=[np.number]).columns.difference(time_columns)
    
    print(f"Applying filter to these numeric columns: {numeric_columns.tolist()}")
    
    # Filter each accelerometer axis
    for column in numeric_columns:
        # Apply zero-phase filter using filtfilt
        filtered_data[column] = signal.filtfilt(b, a, data[column])
    
    return filtered_data

# Apply PCA
def apply_pca(filtered_data, n_components=3):
    print(f"Applying PCA with {n_components} components")
    
    # Select only numeric columns for PCA
    # Exclude time-related columns
    time_columns = [col for col in filtered_data.columns if col.lower() in ['time', 'timestamp', 'datetime']]
    accel_columns = filtered_data.select_dtypes(include=[np.number]).columns.difference(time_columns)
    
    print(f"Using these columns for PCA: {accel_columns.tolist()}")
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data[accel_columns])
    
    # Apply PCA
    n_components = min(n_components, len(accel_columns), filtered_data.shape[0])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create DataFrame with principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
    )
    
    # Add time column if it exists in original data
    for time_col in time_columns:
        pca_df[time_col] = filtered_data[time_col].values
    
    # Show explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance):.2f}")
    
    return pca_df, pca

# Visualize results
def visualize_results(original_data, filtered_data, pca_data, output_dir, test_type, participant_info):
    print(f"Generating visualizations for {test_type}")
    
    # Create descriptive title with participant info
    participant_title = f"{participant_info['name']} ({participant_info['gender']}, {participant_info['age']}y, {participant_info['height']}cm, {participant_info['weight']}kg)"
    
    # Identify numeric columns (likely accelerometer data)
    time_columns = [col for col in original_data.columns if col.lower() in ['time', 'timestamp', 'datetime']]
    accel_columns = original_data.select_dtypes(include=[np.number]).columns.difference(time_columns)
    
    # Use the first time column if available, otherwise use index
    x_axis = time_columns[0] if time_columns else None
    
    # Plot original vs filtered data
    fig, axes = plt.subplots(min(3, len(accel_columns)), 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{test_type.upper()} - {participant_title}", fontsize=14)
    
    # Handle case where there's only one subplot
    if len(accel_columns) <= 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for i, col in enumerate(accel_columns[:3]):  # Plot first 3 axes
        if i >= len(axes):
            break
            
        if x_axis:
            axes[i].plot(original_data[x_axis], original_data[col], 'b-', alpha=0.5, label='Original')
            axes[i].plot(filtered_data[x_axis], filtered_data[col], 'r-', label='Filtered')
            axes[i].set_xlabel(x_axis)
        else:
            axes[i].plot(original_data.index, original_data[col], 'b-', alpha=0.5, label='Original')
            axes[i].plot(filtered_data.index, filtered_data[col], 'r-', label='Filtered')
            axes[i].set_xlabel('Sample')
            
        axes[i].set_ylabel(f'{col}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test_type}_original_vs_filtered.png'))
    
    # Plot PCA components
    plt.figure(figsize=(12, 6))
    pc_columns = [col for col in pca_data.columns if col.startswith('PC')]
    
    plt.title(f"{test_type.upper()} - Principal Components - {participant_title}")
    
    if x_axis and x_axis in pca_data.columns:
        for i, pc_col in enumerate(pc_columns[:3]):  # Plot first 3 PCs
            plt.plot(pca_data[x_axis], pca_data[pc_col], label=pc_col)
        plt.xlabel(x_axis)
    else:
        for i, pc_col in enumerate(pc_columns[:3]):  # Plot first 3 PCs
            plt.plot(pca_data.index, pca_data[pc_col], label=pc_col)
        plt.xlabel('Sample')
    
    plt.ylabel('Principal Component Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test_type}_pca_components.png'))
    
    # Close all figures to free memory
    plt.close('all')

# Calculate test metrics from PCA data
def calculate_test_metrics(pca_data):
    """
    Calculate metrics from PCA data to quantify test performance
    """
    metrics = {}
    
    # Basic statistics on principal components
    pc_columns = [col for col in pca_data.columns if col.startswith('PC')]
    
    for pc in pc_columns:
        # Calculate various metrics for each principal component
        metrics[f"{pc}_mean"] = pca_data[pc].mean()
        metrics[f"{pc}_std"] = pca_data[pc].std()
        metrics[f"{pc}_range"] = pca_data[pc].max() - pca_data[pc].min()
        metrics[f"{pc}_iqr"] = pca_data[pc].quantile(0.75) - pca_data[pc].quantile(0.25)
        
        # Calculate frequency domain metrics if there are enough data points
        if len(pca_data) > 10:
            # Simple spectral analysis
            ps = np.abs(np.fft.fft(pca_data[pc].values))**2
            metrics[f"{pc}_spectral_energy"] = np.sum(ps)
            
    # Calculate cross-component metrics if we have multiple PCs
    if len(pc_columns) >= 2:
        for i in range(len(pc_columns)-1):
            for j in range(i+1, len(pc_columns)):
                pc_i = pc_columns[i]
                pc_j = pc_columns[j]
                
                # Correlation between components
                metrics[f"{pc_i}_{pc_j}_corr"] = pca_data[pc_i].corr(pca_data[pc_j])
    
    return metrics

# Function to load participant ratings data
def load_ratings_data(participant_folder):
    """
    Look for and load ratings data in the participant folder
    Expected format: JSON or CSV with ratings information
    """
    ratings_data = {
        'initial_rating': None,
        'final_rating': None,
        'rejected_pair_rating_diff': None,
        'chosen_pair_rating_diff': None,
        'computer_pair_rating_diff': None,
        'timeout_count': None
    }
    
    # Search for ratings file
    ratings_file = None
    for file in os.listdir(participant_folder):
        file_lower = file.lower()
        if 'rating' in file_lower or 'score' in file_lower or 'result' in file_lower:
            ratings_file = os.path.join(participant_folder, file)
            break
    
    if not ratings_file:
        print(f"No ratings file found in {participant_folder}")
        return ratings_data
    
    try:
        # Attempt to load ratings
        _, ext = os.path.splitext(ratings_file)
        if ext.lower() == '.json':
            with open(ratings_file, 'r') as f:
                data = json.load(f)
                
            # Try to map fields in the JSON to our expected ratings fields
            for key in ratings_data:
                if key in data:
                    ratings_data[key] = data[key]
                else:
                    # Try alternative field names
                    alt_names = {
                        'initial_rating': ['initial', 'start_rating', 'start'],
                        'final_rating': ['final', 'end_rating', 'end'],
                        'rejected_pair_rating_diff': ['rejected', 'rejected_diff', 'rejected_pair'],
                        'chosen_pair_rating_diff': ['chosen', 'chosen_diff', 'chosen_pair'],
                        'computer_pair_rating_diff': ['computer', 'computer_diff', 'computer_pair'],
                        'timeout_count': ['timeout', 'timeouts', 'time_out']
                    }
                    
                    found = False
                    for alt in alt_names.get(key, []):
                        if alt in data:
                            ratings_data[key] = data[alt]
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: Could not find '{key}' in ratings data")
        
        elif ext.lower() in ['.csv', '.txt']:
            df = pd.read_csv(ratings_file)
            # Try to map columns to our expected ratings fields
            for key in ratings_data:
                if key in df.columns:
                    # Just take the first value if there are multiple rows
                    ratings_data[key] = df[key].iloc[0]
                    
    except Exception as e:
        print(f"Error loading ratings data: {e}")
    
    return ratings_data

# Process a single participant
def process_participant(participant_folder_path, base_output_dir, sampling_freq=100):
    print(f"\n{'='*50}")
    print(f"Processing participant: {os.path.basename(participant_folder_path)}")
    print(f"{'='*50}")
    
    try:
        # Create output directories
        participant_name = os.path.basename(participant_folder_path)
        output_dir = os.path.join(base_output_dir, participant_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
        
        # List data files for the participant
        data_files = list_participant_data_files(participant_folder_path)
        
        if not data_files:
            print(f"No data files found for {participant_name}. Skipping.")
            return None
        
        # Initialize participant data dictionary
        participant_data = {
            'name': participant_name,
            'gender': None,
            'age': None,
            'weight': None,
            'height': None,
            'test_metrics': {
                PRE_ITUG: None,
                POST_ITUG: None,
                PRE_ISWAY: None,
                POST_ISWAY: None
            },
            'ratings': {},
            'raw_files': {}
        }
        
        # Load ratings data if available
        ratings_data = load_ratings_data(participant_folder_path)
        participant_data['ratings'] = ratings_data
        
        # Process each data file for the participant
        for file_name in data_files:
            file_path = os.path.join(participant_folder_path, file_name)
            print(f"\nProcessing file: {file_name}")
            
            # Parse participant info from filename if not already set
            if participant_data['age'] is None:
                info = parse_participant_info(file_name)
                if info:
                    participant_data['name'] = info['name']
                    participant_data['gender'] = info['gender']
                    participant_data['age'] = info['age']
                    participant_data['weight'] = info['weight']
                    participant_data['height'] = info['height']
            
            # Identify test type
            test_type = identify_test_type(file_name)
            if test_type is None:
                print(f"Skipping file with unknown test type: {file_name}")
                continue
                
            participant_data['raw_files'][test_type] = file_name
            
            try:
                # Load data
                data = load_data(file_path)
                
                # Handle missing values
                processed_data = preprocess_data(data)
                
                # Apply Butterworth filter
                filtered_data = apply_butterworth_filter(processed_data, sampling_freq)
                
                # Apply PCA
                pca_data, pca_model = apply_pca(filtered_data)
                
                # Create test-specific output directory
                test_output_dir = os.path.join(output_dir, test_type)
                os.makedirs(test_output_dir, exist_ok=True)
                
                # Visualize results
                visualize_results(processed_data, filtered_data, pca_data, test_output_dir, 
                                 test_type, {'name': participant_data['name'], 
                                             'gender': participant_data['gender'],
                                             'age': participant_data['age'],
                                             'weight': participant_data['weight'], 
                                             'height': participant_data['height']})
                
                # Calculate test metrics
                metrics = calculate_test_metrics(pca_data)
                participant_data['test_metrics'][test_type] = metrics
                
                # Save processed data
                filtered_data_file = os.path.join(test_output_dir, 'filtered_data.csv')
                pca_data_file = os.path.join(test_output_dir, 'pca_data.csv')
                metrics_file = os.path.join(test_output_dir, 'metrics.json')
                
                filtered_data.to_csv(filtered_data_file, index=False)
                pca_data.to_csv(pca_data_file, index=False)
                
                # Save metrics to JSON
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"Saved filtered data to: {filtered_data_file}")
                print(f"Saved PCA data to: {pca_data_file}")
                print(f"Saved metrics to: {metrics_file}")
                
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"\nCompleted processing participant: {participant_name}")
        
        # Save the complete participant data summary
        summary_file = os.path.join(output_dir, 'participant_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(participant_data, f, indent=2)
        
        return participant_data
        
    except Exception as e:
        print(f"Error processing participant {participant_folder_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to create aggregate summary across all participants
def create_aggregate_summary(all_participant_data, output_dir):
    print("\nCreating aggregate summary across all participants...")
    
    # Create a DataFrame for the summary
    summary_columns = [
        'name', 'gender', 'age', 'weight', 'height',
        'pre_itug_completeness', 'post_itug_completeness', 
        'pre_isway_completeness', 'post_isway_completeness',
        'initial_rating', 'final_rating', 
        'rejected_pair_rating_diff', 'chosen_pair_rating_diff',
        'computer_pair_rating_diff', 'timeout_count'
    ]
    
    # Add key metrics from each test (first PC only for summary)
    for test in [PRE_ITUG, POST_ITUG, PRE_ISWAY, POST_ISWAY]:
        summary_columns.extend([
            f'{test}_pc1_mean', f'{test}_pc1_std', f'{test}_pc1_range'
        ])
    
    summary_df = pd.DataFrame(columns=summary_columns)
    
    # Populate the summary DataFrame
    for participant_data in all_participant_data:
        if participant_data is None:
            continue
            
        row = {
            'name': participant_data['name'],
            'gender': participant_data['gender'],
            'age': participant_data['age'],
            'weight': participant_data['weight'],
            'height': participant_data['height'],
            'initial_rating': participant_data['ratings'].get('initial_rating'),
            'final_rating': participant_data['ratings'].get('final_rating'),
            'rejected_pair_rating_diff': participant_data['ratings'].get('rejected_pair_rating_diff'),
            'chosen_pair_rating_diff': participant_data['ratings'].get('chosen_pair_rating_diff'),
            'computer_pair_rating_diff': participant_data['ratings'].get('computer_pair_rating_diff'),
            'timeout_count': participant_data['ratings'].get('timeout_count')
        }
        
        # Add test completeness indicators
        for test in [PRE_ITUG, POST_ITUG, PRE_ISWAY, POST_ISWAY]:
            row[f'{test}_completeness'] = 1 if participant_data['test_metrics'][test] else 0
        
        # Add key metrics from each test
        for test in [PRE_ITUG, POST_ITUG, PRE_ISWAY, POST_ISWAY]:
            metrics = participant_data['test_metrics'][test]
            if metrics:
                row[f'{test}_pc1_mean'] = metrics.get('PC1_mean')
                row[f'{test}_pc1_std'] = metrics.get('PC1_std')
                row[f'{test}_pc1_range'] = metrics.get('PC1_range')
        
        # Add row to summary DataFrame
        summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
    
    # Save summary to CSV
    summary_file = os.path.join(output_dir, 'all_participants_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved aggregate summary to: {summary_file}")
    
    # Generate some basic statistics on the cohort
    print("\nCohort Statistics:")
    print(f"Total participants: {len(summary_df)}")
    print(f"Gender distribution: {summary_df['gender'].value_counts().to_dict()}")
    print(f"Age range: {summary_df['age'].min()} - {summary_df['age'].max()} years (mean: {summary_df['age'].mean():.1f})")
    print(f"Weight range: {summary_df['weight'].min()} - {summary_df['weight'].max()} kg (mean: {summary_df['weight'].mean():.1f})")
    print(f"Height range: {summary_df['height'].min()} - {summary_df['height'].max()} cm (mean: {summary_df['height'].mean():.1f})")
    
    # Generate completeness statistics
    for test in [PRE_ITUG, POST_ITUG, PRE_ISWAY, POST_ISWAY]:
        complete_count = summary_df[f'{test}_completeness'].sum()
        print(f"{test.upper()} test completion: {complete_count}/{len(summary_df)} ({complete_count/len(summary_df)*100:.1f}%)")
    
    # Create some visualizations of the aggregate data
    create_aggregate_visualizations(summary_df, output_dir)
    
    return summary_df

# Create visualizations for aggregate data
def create_aggregate_visualizations(summary_df, output_dir):
    print("\nCreating aggregate visualizations...")
    
    # Age distribution by gender
    plt.figure(figsize=(10, 6))
    summary_df.boxplot(column='age', by='gender')
    plt.title('Age Distribution by Gender')
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution_by_gender.png'))
    
    # Height and weight scatter plot by gender
    plt.figure(figsize=(10, 6))
    for gender, group in summary_df.groupby('gender'):
        plt.scatter(group['height'], group['weight'], label=gender, alpha=0.7)
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.title('Height vs Weight by Gender')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'height_vs_weight_by_gender.png'))
    
    # Pre vs Post test metrics comparison (if available)
    # For example, comparing pre and post ITUG PC1 mean
    if 'pre_itug_pc1_mean' in summary_df.columns and 'post_itug_pc1_mean' in summary_df.columns:
        # Remove rows with NaN values for these columns
        valid_df = summary_df.dropna(subset=['pre_itug_pc1_mean', 'post_itug_pc1_mean'])
        
        if len(valid_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(valid_df['pre_itug_pc1_mean'], valid_df['post_itug_pc1_mean'])
            min_val = min(valid_df['pre_itug_pc1_mean'].min(), valid_df['post_itug_pc1_mean'].min())
            max_val = max(valid_df['pre_itug_pc1_mean'].max(), valid_df['post_itug_pc1_mean'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)  # Identity line
            plt.xlabel('Pre ITUG PC1 Mean')
            plt.ylabel('Post ITUG PC1 Mean')
            plt.title('Pre vs Post ITUG Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pre_vs_post_itug_comparison.png'))
    
    # Similarly for ISWAY data
    if 'pre_isway_pc1_mean' in summary_df.columns and 'post_isway_pc1_mean' in summary_df.columns:
        valid_df = summary_df.dropna(subset=['pre_isway_pc1_mean', 'post_isway_pc1_mean'])
        
        if len(valid_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(valid_df['pre_isway_pc1_mean'], valid_df['post_isway_pc1_mean'])
            min_val = min(valid_df['pre_isway_pc1_mean'].min(), valid_df['post_isway_pc1_mean'].min())
            max_val = max(valid_df['pre_isway_pc1_mean'].max(), valid_df['post_isway_pc1_mean'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)  # Identity line
            plt.xlabel('Pre ISWAY PC1 Mean')
            plt.ylabel('Post ISWAY PC1 Mean')
            plt.title('Pre vs Post ISWAY Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pre_vs_post_isway_comparison.png'))
    
    # Rating changes visualization (initial vs final)
    if 'initial_rating' in summary_df.columns and 'final_rating' in summary_df.columns:
        valid_df = summary_df.dropna(subset=['initial_rating', 'final_rating'])
        
        if len(valid_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(valid_df['initial_rating'], valid_df['final_rating'])
            min_val = min(valid_df['initial_rating'].min(), valid_df['final_rating'].min())
            max_val = max(valid_df['initial_rating'].max(), valid_df['final_rating'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)  # Identity line
            plt.xlabel('Initial Rating')
            plt.ylabel('Final Rating')
            plt.title('Initial vs Final Rating Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'initial_vs_final_rating.png'))
    
    # Close all figures
    plt.close('all')

def main():
    print("================================")
    print("Accelerometer Data Analysis Tool")
    print("================================")
    print("This script analyzes ITUG and ISWAY tests, both pre and post, and organizes data by participant")
    
    try:
        # Define the base directory containing participant folders
        base_dir = "18-04-2025SENSORDATA"  # Default name
        
        # Check if base directory exists; if not, look in current directory or ask
        if not os.path.exists(base_dir):
            # Try to find it in current directory
            current_dir = os.getcwd()
            possible_dirs = [d for d in os.listdir(current_dir) 
                            if os.path.isdir(d) and '2025' in d and 'SENSOR' in d.upper()]
            
            if possible_dirs:
                if len(possible_dirs) == 1:
                    base_dir = possible_dirs[0]
                    print(f"Found sensor data directory: {base_dir}")
                else:
                    print("Multiple possible sensor data directories found:")
                    for i, d in enumerate(possible_dirs):
                        print(f"{i+1}. {d}")
                    choice = input("Enter the number of the correct directory: ")
                    base_dir = possible_dirs[int(choice)-1]
            else:
                base_dir = input("Enter the path to the directory containing participant folders: ")
                if not base_dir:
                    base_dir = "."  # Default to current directory
        
        # Create output directory
        output_dir = os.path.join(base_dir, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
        
        # Set sampling frequency (Hz)
        sampling_freq = 100  # Default for typical accelerometer data
        
        # Ask for sampling frequency
        try:
            user_freq = input(f"Enter sampling frequency in Hz (default: {sampling_freq}): ")
            if user_freq.strip():
                sampling_freq = float(user_freq)
        except:
            print(f"Using default sampling frequency: {sampling_freq} Hz")
        
        # List all participant folders
        participant_folders = list_participant_folders(base_dir)
        
        if not participant_folders:
            print("No participant folders found.")
            return
        
        # Process each participant and collect their data
        all_participant_data = []
        successful_participants = 0
        
        for folder in participant_folders:
            folder_path = os.path.join(base_dir, folder)
            participant_data = process_participant(folder_path, output_dir, sampling_freq)
            
            if participant_data:
                all_participant_data.append(participant_data)
                successful_participants += 1
        
        print(f"\nComplete! Successfully processed {successful_participants} out of {len(participant_folders)} participants.")
        
        # Create aggregate summary if we have data
        if all_participant_data:
            summary_df = create_aggregate_summary(all_participant_data, output_dir)
            
            # Save the full raw data structure for potential further processing
            with open(os.path.join(output_dir, 'all_participant_raw_data.json'), 'w') as f:
                json.dump(all_participant_data, f, indent=2)
                
            print(f"\nSummary statistics:")
            print(f"Total participants: {len(summary_df)}")
            print(f"Gender distribution: {summary_df['gender'].value_counts().to_dict()}")
            print(f"Age: mean={summary_df['age'].mean():.1f}, min={summary_df['age'].min()}, max={summary_df['age'].max()}")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()