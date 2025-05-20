import os
import json
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
import pingouin as pg
from sklearn.linear_model import LinearRegression
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set output directory for results
output_dir = '/Users/norachuw/PycharmProjects/sensordata/analysis_results/statistical_results'
os.makedirs(output_dir, exist_ok=True)

# Path pattern for participant data files
base_path = '/Users/norachuw/PycharmProjects/sensordata/analysis_results/sensor_analysis'
pattern = os.path.join(base_path, '*', 'participant_summary.json')

# Get all participant data files
participant_files = glob.glob(pattern)
print(f"Found {len(participant_files)} participant files.")

# Load all participant data into a list
participants_data = []
for file_path in participant_files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            participants_data.append(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

print(f"Successfully loaded data for {len(participants_data)} participants.")


# Convert participant data to DataFrame for analysis
def create_participant_df(participants_data):
    df_list = []

    for participant in participants_data:
        # Basic demographic data
        participant_row = {
            'name': participant.get('name', ''),
            'gender': participant.get('gender', ''),
            'gender_code': participant.get('gender_code', ''),
            'age': participant.get('age', np.nan),
            'weight': participant.get('weight', np.nan),
            'height': participant.get('height', np.nan),
            'bmi': participant.get('bmi', np.nan),
        }

        # Pre-ITUG metrics
        if 'pre_itug_metrics' in participant and participant['pre_itug_metrics'].get('valid', False):
            pre_itug = participant['pre_itug_metrics']
            participant_row.update({
                'pre_itug_duration': pre_itug.get('duration_seconds', np.nan),
                'pre_itug_PC1_mean': pre_itug.get('PC1_mean', np.nan),
                'pre_itug_PC1_std': pre_itug.get('PC1_std', np.nan),
                'pre_itug_PC1_range': pre_itug.get('PC1_range', np.nan),
                'pre_itug_PC1_dominant_freq': pre_itug.get('PC1_dominant_freq', np.nan),
                'pre_itug_stride_frequency': pre_itug.get('stride_frequency', np.nan),
                'pre_itug_num_strides': pre_itug.get('num_strides', np.nan),
            })

        # Pre-ISWAY metrics
        if 'pre_isway_metrics' in participant and participant['pre_isway_metrics'].get('valid', False):
            pre_isway = participant['pre_isway_metrics']
            participant_row.update({
                'pre_isway_PC1_mean': pre_isway.get('PC1_mean', np.nan),
                'pre_isway_PC1_std': pre_isway.get('PC1_std', np.nan),
                'pre_isway_PC1_range': pre_isway.get('PC1_range', np.nan),
                'pre_isway_PC1_dominant_freq': pre_isway.get('PC1_dominant_freq', np.nan),
                'pre_isway_PC1_spectral_entropy': pre_isway.get('PC1_spectral_entropy', np.nan),
                'pre_isway_sway_area': pre_isway.get('sway_area', np.nan),
                'pre_isway_mean_distance': pre_isway.get('mean_distance', np.nan),
                'pre_isway_path_length': pre_isway.get('path_length', np.nan),
            })

        # Post-ITUG metrics
        if 'post_itug_metrics' in participant and participant['post_itug_metrics'].get('valid', False):
            post_itug = participant['post_itug_metrics']
            participant_row.update({
                'post_itug_duration': post_itug.get('duration_seconds', np.nan),
                'post_itug_PC1_mean': post_itug.get('PC1_mean', np.nan),
                'post_itug_PC1_std': post_itug.get('PC1_std', np.nan),
                'post_itug_PC1_range': post_itug.get('PC1_range', np.nan),
                'post_itug_PC1_dominant_freq': post_itug.get('PC1_dominant_freq', np.nan),
                'post_itug_stride_frequency': post_itug.get('stride_frequency', np.nan),
                'post_itug_num_strides': post_itug.get('num_strides', np.nan),
            })

        # Post-ISWAY metrics
        if 'post_isway_metrics' in participant and participant['post_isway_metrics'].get('valid', False):
            post_isway = participant['post_isway_metrics']
            participant_row.update({
                'post_isway_PC1_mean': post_isway.get('PC1_mean', np.nan),
                'post_isway_PC1_std': post_isway.get('PC1_std', np.nan),
                'post_isway_PC1_range': post_isway.get('PC1_range', np.nan),
                'post_isway_PC1_dominant_freq': post_isway.get('PC1_dominant_freq', np.nan),
                'post_isway_PC1_spectral_entropy': post_isway.get('PC1_spectral_entropy', np.nan),
                'post_isway_sway_area': post_isway.get('sway_area', np.nan),
                'post_isway_mean_distance': post_isway.get('mean_distance', np.nan),
                'post_isway_path_length': post_isway.get('path_length', np.nan),
            })

        # Food rating experiment metrics
        participant_row.update({
            'initial_rating': participant.get('initial_rating', np.nan),
            'final_rating': participant.get('final_rating', np.nan),
            'rejected_pair_rating_diff': participant.get('rejected_pair_rating_diff', np.nan),
            'chosen_pair_rating_diff': participant.get('chosen_pair_rating_diff', np.nan),
            'computer_pair_rating_diff': participant.get('computer_pair_rating_diff', np.nan),
            'timeout_count': participant.get('timeout_count', np.nan),
        })

        df_list.append(participant_row)

    return pd.DataFrame(df_list)


# Create the main DataFrame
df = create_participant_df(participants_data)
print(f"DataFrame created with {df.shape[0]} rows and {df.shape[1]} columns.")


# Calculate change scores for balance measures
def add_change_scores(df):
    # ITUG change scores
    itug_metrics = ['duration', 'PC1_mean', 'PC1_std', 'PC1_range',
                    'PC1_dominant_freq', 'stride_frequency', 'num_strides']

    for metric in itug_metrics:
        df[f'delta_itug_{metric}'] = df[f'post_itug_{metric}'] - df[f'pre_itug_{metric}']

    # ISWAY change scores
    isway_metrics = ['PC1_mean', 'PC1_std', 'PC1_range', 'PC1_dominant_freq',
                     'PC1_spectral_entropy', 'sway_area', 'mean_distance', 'path_length']

    for metric in isway_metrics:
        df[f'delta_isway_{metric}'] = df[f'post_isway_{metric}'] - df[f'pre_isway_{metric}']

    # Calculate cognitive dissonance measure (final - initial rating)
    df['rating_change'] = df['final_rating'] - df['initial_rating']

    return df


df = add_change_scores(df)
print("Change scores calculated and added to DataFrame.")

# Save the processed DataFrame
df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
print(f"Processed data saved to {os.path.join(output_dir, 'processed_data.csv')}")


# ----------------------------------------------------------------------------------
# 1. Repeated Measures Correlation Analysis
# ----------------------------------------------------------------------------------
def perform_repeated_measures_correlation(df):
    results = {}

    # Define cognitive dissonance measures
    cognitive_dissonance_measures = [
        'rating_change',
        'rejected_pair_rating_diff',
        'chosen_pair_rating_diff',
        'computer_pair_rating_diff',
        'initial_rating',
        'final_rating'
    ]

    # Define balance measures
    pre_itug_measures = ['pre_itug_duration', 'pre_itug_PC1_std', 'pre_itug_PC1_range',
                         'pre_itug_PC1_dominant_freq', 'pre_itug_stride_frequency', 'pre_itug_num_strides']

    pre_isway_measures = ['pre_isway_PC1_std', 'pre_isway_PC1_range', 'pre_isway_PC1_dominant_freq',
                          'pre_isway_PC1_spectral_entropy', 'pre_isway_sway_area',
                          'pre_isway_mean_distance', 'pre_isway_path_length']

    # Correlate pre-balance measures with cognitive dissonance measures
    for balance_measure in pre_itug_measures + pre_isway_measures:
        for cd_measure in cognitive_dissonance_measures:
            # Filter out missing values
            valid_data = df[[balance_measure, cd_measure]].dropna()

            if len(valid_data) > 5:  # Ensure enough data points
                correlation, p_value = stats.pearsonr(valid_data[balance_measure], valid_data[cd_measure])
                results[f"{balance_measure}_vs_{cd_measure}"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "n": len(valid_data)
                }

    # Save results to file
    with open(os.path.join(output_dir, 'repeated_measures_correlation.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Create a summary DataFrame
    corr_df = pd.DataFrame([
        {
            'Balance_Measure': k.split('_vs_')[0],
            'Cognitive_Measure': k.split('_vs_')[1],
            'Correlation': v['correlation'],
            'P_Value': v['p_value'],
            'Sample_Size': v['n']
        }
        for k, v in results.items()
    ])

    # Save as CSV
    corr_df.to_csv(os.path.join(output_dir, 'repeated_measures_correlation.csv'), index=False)

    # Create visualizations
    plt.figure(figsize=(14, 10))

    # Create a pivot table for the heatmap
    heatmap_data = corr_df.pivot(index='Balance_Measure', columns='Cognitive_Measure', values='Correlation')

    # Create a heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation between Balance Measures and Cognitive Dissonance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))

    return results


print("Performing repeated measures correlation analysis...")
corr_results = perform_repeated_measures_correlation(df)


# ----------------------------------------------------------------------------------
# 2. Linear Regression Analysis
# ----------------------------------------------------------------------------------
def perform_linear_regression(df):
    results = {}

    # Print original data types to diagnose issues
    print("Original data types:")
    print(df.dtypes)

    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # Convert any potential string columns to numeric, coercing errors to NaN
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' and col != 'gender_code':  # Skip gender which is categorical
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    print("Data types after conversion:")
    print(df_clean.dtypes)

    # Define cognitive dissonance measures as dependent variables
    dependent_vars = [
        'rating_change',
        'rejected_pair_rating_diff',
        'chosen_pair_rating_diff',
        'computer_pair_rating_diff'
    ]

    # Define groups of independent variables (predictors)
    predictor_groups = {
        'demographic': ['age', 'gender_code', 'bmi'],
        'pre_itug': ['pre_itug_duration', 'pre_itug_PC1_std', 'pre_itug_PC1_range',
                     'pre_itug_PC1_dominant_freq', 'pre_itug_stride_frequency', 'pre_itug_num_strides'],
        'pre_isway': ['pre_isway_PC1_std', 'pre_isway_PC1_range', 'pre_isway_PC1_dominant_freq',
                      'pre_isway_PC1_spectral_entropy', 'pre_isway_sway_area',
                      'pre_isway_mean_distance', 'pre_isway_path_length']
    }

    # Run regression models
    for dep_var in dependent_vars:
        for group_name, predictors in predictor_groups.items():
            # Handle categorical predictors
            if 'gender_code' in predictors:
                # Create dummy variables for gender
                df_model = pd.get_dummies(df_clean[predictors + [dep_var]].dropna(), columns=['gender_code'],
                                          drop_first=True)
                # Update predictors list to include dummy variables instead of original categorical variable
                cat_cols = [col for col in df_model.columns if col.startswith('gender_code_')]
                model_predictors = [p for p in predictors if p != 'gender_code'] + cat_cols
            else:
                df_model = df_clean[predictors + [dep_var]].dropna()
                model_predictors = predictors

            if len(df_model) > len(model_predictors) + 2:  # Ensure enough data points
                X = df_model[model_predictors]
                y = df_model[dep_var]

                # Add constant for statsmodels
                X_sm = sm.add_constant(X)

                # Verify all data is numeric before regression
                print(f"Model for {dep_var} ~ {group_name}")
                print("X_sm dtypes:", X_sm.dtypes)
                print("y dtype:", y.dtype)
                print("X_sm shape:", X_sm.shape)
                print("y shape:", y.shape)

                # Fit the regression model
                model = sm.OLS(y, X_sm).fit()

                # Store results
                results[f"{dep_var}_predicted_by_{group_name}"] = {
                    "r_squared": model.rsquared,
                    "adj_r_squared": model.rsquared_adj,
                    "f_statistic": model.fvalue,
                    "p_value": model.f_pvalue,
                    "n": len(df_model),
                    "coefficients": {
                        name: {
                            "coef": coef,
                            "p_value": p_value
                        }
                        for name, coef, p_value in zip(model.params.index, model.params, model.pvalues)
                    }
                }

    return results
    # Save results to file
    with open(os.path.join(output_dir, 'linear_regression.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Create a summary table
    regression_summary = []

    for model_name, model_results in results.items():
        # Extract dependent variable and predictor group
        parts = model_name.split('_predicted_by_')
        dep_var = parts[0]
        predictor_group = parts[1]

        # Add overall model statistics
        regression_summary.append({
            'Dependent_Variable': dep_var,
            'Predictor_Group': predictor_group,
            'R_Squared': model_results['r_squared'],
            'Adj_R_Squared': model_results['adj_r_squared'],
            'F_Statistic': model_results['f_statistic'],
            'P_Value': model_results['p_value'],
            'Sample_Size': model_results['n']
        })

    # Convert to DataFrame and save
    regression_df = pd.DataFrame(regression_summary)
    regression_df.to_csv(os.path.join(output_dir, 'linear_regression_summary.csv'), index=False)

    # Create a coefficient table
    coef_rows = []

    for model_name, model_results in results.items():
        # Extract dependent variable and predictor group
        parts = model_name.split('_predicted_by_')
        dep_var = parts[0]
        predictor_group = parts[1]

        # Add each coefficient
        for predictor, coef_info in model_results['coefficients'].items():
            coef_rows.append({
                'Dependent_Variable': dep_var,
                'Predictor_Group': predictor_group,
                'Predictor': predictor,
                'Coefficient': coef_info['coef'],
                'P_Value': coef_info['p_value']
            })

    # Convert to DataFrame and save
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(os.path.join(output_dir, 'linear_regression_coefficients.csv'), index=False)

    # Create visualizations
    plt.figure(figsize=(12, 8))

    # Bar plot for R-squared by predictor group and dependent variable
    sns.barplot(x='Predictor_Group', y='R_Squared', hue='Dependent_Variable', data=regression_df)
    plt.title('R-squared Values by Predictor Group and Dependent Variable')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_rsquared.png'))

    return results


print("Performing linear regression analysis...")
regression_results = perform_linear_regression(df)


def perform_linear_regression(df):
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # Convert any potential string columns to numeric, coercing errors to NaN
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Drop rows with missing values
    df_clean = df_clean.dropna()

    # Check data types after conversion
    print("Data types after conversion:")
    print(df_clean.dtypes)

    # Prepare variables for regression (assuming you know which columns to use)
    # Replace these with your actual feature and target columns
    X = df_clean[['feature1', 'feature2', 'feature3']]  # Update with your feature columns
    y = df_clean['target']  # Update with your target column

    # Add constant
    X_sm = sm.add_constant(X)

    # Verify all data is numeric
    print("X_sm dtypes:", X_sm.dtypes)
    print("y dtype:", y.dtype)

    # If there are still non-numeric values, they'll be visible here
    # print(X_sm.head())
    # print(y.head())

    # Fit the model
    model = sm.OLS(y, X_sm).fit()

    return model

# ----------------------------------------------------------------------------------
# 3. Repeated Measures ANOVA
# ----------------------------------------------------------------------------------
def perform_repeated_measures_anova(df):
    results = {}

    # Define pairs of pre/post measures for ANOVA
    measure_pairs = [
        # ITUG measures
        ('pre_itug_duration', 'post_itug_duration', 'ITUG Duration'),
        ('pre_itug_PC1_std', 'post_itug_PC1_std', 'ITUG PC1 Std'),
        ('pre_itug_PC1_range', 'post_itug_PC1_range', 'ITUG PC1 Range'),
        ('pre_itug_PC1_dominant_freq', 'post_itug_PC1_dominant_freq', 'ITUG PC1 Dominant Freq'),
        ('pre_itug_stride_frequency', 'post_itug_stride_frequency', 'ITUG Stride Frequency'),
        ('pre_itug_num_strides', 'post_itug_num_strides', 'ITUG Num Strides'),

        # ISWAY measures
        ('pre_isway_PC1_std', 'post_isway_PC1_std', 'ISWAY PC1 Std'),
        ('pre_isway_PC1_range', 'post_isway_PC1_range', 'ISWAY PC1 Range'),
        ('pre_isway_PC1_dominant_freq', 'post_isway_PC1_dominant_freq', 'ISWAY PC1 Dominant Freq'),
        ('pre_isway_PC1_spectral_entropy', 'post_isway_PC1_spectral_entropy', 'ISWAY PC1 Spectral Entropy'),
        ('pre_isway_sway_area', 'post_isway_sway_area', 'ISWAY Sway Area'),
        ('pre_isway_mean_distance', 'post_isway_mean_distance', 'ISWAY Mean Distance'),
        ('pre_isway_path_length', 'post_isway_path_length', 'ISWAY Path Length')
    ]

    for pre_measure, post_measure, measure_name in measure_pairs:
        # Prepare data for ANOVA - we need valid pre and post measurements
        valid_data = df[[pre_measure, post_measure]].dropna()

        if len(valid_data) >= 5:  # Ensure enough data points
            # Reshape data for ANOVA (long format)
            long_data = pd.DataFrame({
                'subject': np.repeat(range(len(valid_data)), 2),
                'time': np.tile(['pre', 'post'], len(valid_data)),
                'value': np.concatenate([valid_data[pre_measure].values, valid_data[post_measure].values])
            })

            try:
                # Perform repeated measures ANOVA
                aov = pg.rm_anova(data=long_data, dv='value', within='time', subject='subject')

                # Store results
                results[measure_name] = {
                    'F': aov.loc[0, 'F'],
                    'p-value': aov.loc[0, 'p-unc'],
                    'df1': aov.loc[0, 'ddof1'],
                    'df2': aov.loc[0, 'ddof2'],
                    'n': len(valid_data),
                    'mean_pre': valid_data[pre_measure].mean(),
                    'mean_post': valid_data[post_measure].mean(),
                    'std_pre': valid_data[pre_measure].std(),
                    'std_post': valid_data[post_measure].std()
                }
            except Exception as e:
                print(f"Error performing ANOVA for {measure_name}: {e}")

    # Save results to file
    with open(os.path.join(output_dir, 'repeated_measures_anova.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Create a summary DataFrame
    anova_df = pd.DataFrame([
        {
            'Measure': measure,
            'F_Statistic': result['F'],
            'P_Value': result['p-value'],
            'DF1': result['df1'],
            'DF2': result['df2'],
            'Sample_Size': result['n'],
            'Pre_Mean': result['mean_pre'],
            'Post_Mean': result['mean_post'],
            'Pre_Std': result['std_pre'],
            'Post_Std': result['std_post'],
            'Significant': result['p-value'] < 0.05
        }
        for measure, result in results.items()
    ])

    # Save as CSV
    anova_df.to_csv(os.path.join(output_dir, 'repeated_measures_anova.csv'), index=False)

    # Create visualizations
    plt.figure(figsize=(14, 10))

    # Bar plot for means with error bars
    measure_list = anova_df['Measure'].tolist()
    pre_means = anova_df['Pre_Mean'].tolist()
    post_means = anova_df['Post_Mean'].tolist()
    pre_stds = anova_df['Pre_Std'].tolist()
    post_stds = anova_df['Post_Std'].tolist()

    x = np.arange(len(measure_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 10))
    rects1 = ax.bar(x - width / 2, pre_means, width, label='Pre', yerr=pre_stds, alpha=0.7)
    rects2 = ax.bar(x + width / 2, post_means, width, label='Post', yerr=post_stds, alpha=0.7)

    ax.set_xlabel('Measure')
    ax.set_ylabel('Value')
    ax.set_title('Pre vs Post Measurements')
    ax.set_xticks(x)
    ax.set_xticklabels(measure_list, rotation=45, ha='right')
    ax.legend()

    # Add asterisks for significant differences
    for i, measure in enumerate(measure_list):
        if anova_df.loc[anova_df['Measure'] == measure, 'Significant'].iloc[0]:
            ax.text(i, max(pre_means[i], post_means[i]) + max(pre_stds[i], post_stds[i]), '*',
                    fontsize=16, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anova_pre_post_comparison.png'))

    return results


print("Performing repeated measures ANOVA...")
anova_results = perform_repeated_measures_anova(df)


# ----------------------------------------------------------------------------------
# 4. Paired t-test
# ----------------------------------------------------------------------------------
def perform_paired_ttest(df):
    results = {}

    # Define pairs of pre/post measures for t-test
    measure_pairs = [
        # ITUG measures
        ('pre_itug_duration', 'post_itug_duration', 'ITUG Duration'),
        ('pre_itug_PC1_std', 'post_itug_PC1_std', 'ITUG PC1 Std'),
        ('pre_itug_PC1_range', 'post_itug_PC1_range', 'ITUG PC1 Range'),
        ('pre_itug_PC1_dominant_freq', 'post_itug_PC1_dominant_freq', 'ITUG PC1 Dominant Freq'),
        ('pre_itug_stride_frequency', 'post_itug_stride_frequency', 'ITUG Stride Frequency'),
        ('pre_itug_num_strides', 'post_itug_num_strides', 'ITUG Num Strides'),

        # ISWAY measures
        ('pre_isway_PC1_std', 'post_isway_PC1_std', 'ISWAY PC1 Std'),
        ('pre_isway_PC1_range', 'post_isway_PC1_range', 'ISWAY PC1 Range'),
        ('pre_isway_PC1_dominant_freq', 'post_isway_PC1_dominant_freq', 'ISWAY PC1 Dominant Freq'),
        ('pre_isway_PC1_spectral_entropy', 'post_isway_PC1_spectral_entropy', 'ISWAY PC1 Spectral Entropy'),
        ('pre_isway_sway_area', 'post_isway_sway_area', 'ISWAY Sway Area'),
        ('pre_isway_mean_distance', 'post_isway_mean_distance', 'ISWAY Mean Distance'),
        ('pre_isway_path_length', 'post_isway_path_length', 'ISWAY Path Length')
    ]

    for pre_measure, post_measure, measure_name in measure_pairs:
        # Prepare data for t-test - we need valid pre and post measurements
        valid_data = df[[pre_measure, post_measure]].dropna()

        if len(valid_data) >= 5:  # Ensure enough data points
            try:
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(valid_data[pre_measure], valid_data[post_measure])

                # Calculate Cohen's d for effect size
                d = (valid_data[post_measure].mean() - valid_data[pre_measure].mean()) / np.sqrt(
                    (valid_data[pre_measure].std() ** 2 + valid_data[post_measure].std() ** 2) / 2)

                # Store results
                results[measure_name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': d,
                    'n': len(valid_data),
                    'mean_pre': valid_data[pre_measure].mean(),
                    'mean_post': valid_data[post_measure].mean(),
                    'std_pre': valid_data[pre_measure].std(),
                    'std_post': valid_data[post_measure].std(),
                    'mean_diff': valid_data[post_measure].mean() - valid_data[pre_measure].mean()
                }
            except Exception as e:
                print(f"Error performing t-test for {measure_name}: {e}")

    # Save results to file
    with open(os.path.join(output_dir, 'paired_ttest.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Create a summary DataFrame
    ttest_df = pd.DataFrame([
        {
            'Measure': measure,
            'T_Statistic': result['t_statistic'],
            'P_Value': result['p_value'],
            'Cohens_d': result['cohens_d'],
            'Sample_Size': result['n'],
            'Pre_Mean': result['mean_pre'],
            'Post_Mean': result['mean_post'],
            'Mean_Difference': result['mean_diff'],
            'Significant': result['p_value'] < 0.05
        }
        for measure, result in results.items()
    ])

    # Save as CSV
    ttest_df.to_csv(os.path.join(output_dir, 'paired_ttest.csv'), index=False)

    # Create visualizations
    plt.figure(figsize=(12, 8))

    # Create a horizontal bar plot of mean differences
    significant_mask = ttest_df['Significant']

    plt.figure(figsize=(10, 8))
    bars = plt.barh(ttest_df['Measure'], ttest_df['Mean_Difference'])

    # Color bars based on significance
    for i, bar in enumerate(bars):
        if ttest_df.iloc[i]['Significant']:
            bar.set_color('red')
        else:
            bar.set_color('gray')

    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Mean Difference (Post - Pre)')
    plt.title('Mean Differences Between Pre and Post Measurements')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttest_mean_differences.png'))

    return results


print("Performing paired t-tests...")
ttest_results = perform_paired_ttest(df)


# ----------------------------------------------------------------------------------
# 5. Change Score Analysis
# ----------------------------------------------------------------------------------
def perform_change_score_analysis(df):
    results = {}

    # Define cognitive dissonance measures
    cognitive_dissonance_measures = [
        'rating_change',
        'rejected_pair_rating_diff',
        'chosen_pair_rating_diff',
        'computer_pair_rating_diff',
        'initial_rating',
        'final_rating'
    ]

    # Define balance change measures
    balance_changes = [
        # ITUG changes
        'delta_itug_duration',
        'delta_itug_PC1_mean',
        'delta_itug_PC1_std',
        'delta_itug_PC1_range',
        'delta_itug_PC1_dominant_freq',
        'delta_itug_stride_frequency',
        'delta_itug_num_strides',

        # ISWAY changes
        'delta_isway_PC1_mean',
        'delta_isway_PC1_std',
        'delta_isway_PC1_range',
        'delta_isway_PC1_dominant_freq',
        'delta_isway_PC1_spectral_entropy',
        'delta_isway_sway_area',
        'delta_isway_mean_distance',
        'delta_isway_path_length'
    ]

    # Correlate balance change measures with cognitive dissonance measures
    for change_measure in balance_changes:
        for cd_measure in cognitive_dissonance_measures:
            # Filter out missing values
            valid_data = df[[change_measure, cd_measure]].dropna()

            if len(valid_data) > 5:  # Ensure enough data points
                correlation, p_value = stats.pearsonr(valid_data[change_measure], valid_data[cd_measure])
                results[f"{change_measure}_vs_{cd_measure}"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "n": len(valid_data)
                }

    # Save results to file
    with open(os.path.join(output_dir, 'change_score_analysis.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Create a summary DataFrame
    change_df = pd.DataFrame([
        {
            'Balance_Change_Measure': k.split('_vs_')[0],
            'Cognitive_Measure': k.split('_vs_')[1],
            'Correlation': v['correlation'],
            'P_Value': v['p_value'],
            'Sample_Size': v['n'],
            'Significant': v['p_value'] < 0.05
        }
        for k, v in results.items()
    ])

    # Save as CSV
    change_df.to_csv(os.path.join(output_dir, 'change_score_analysis.csv'), index=False)

    # Create visualizations
    plt.figure(figsize=(14, 10))

    # Create a pivot table for the heatmap
    heatmap_data = change_df.pivot(index='Balance_Change_Measure', columns='Cognitive_Measure', values='Correlation')

    # Create a heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation between Balance Changes and Cognitive Dissonance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'change_score_correlation_heatmap.png'))

    # Scatter plots for significant correlations
    significant_corrs = change_df[change_df['Significant']]

    if len(significant_corrs) > 0:
        # Create up to 9 of the most significant correlations
        top_significant = significant_corrs.sort_values(by='P_Value').head(min(9, len(significant_corrs)))

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        for i, (_, row) in enumerate(top_significant.iterrows()):
            if i >= 9:  # Limit to 9 plots
                break

            balance_measure = row['Balance_Change_Measure']
            cd_measure = row['Cognitive_Measure']

            valid_data = df[[balance_measure, cd_measure]].dropna()

            ax = axes[i]
            sns.regplot(x=balance_measure, y=cd_measure, data=valid_data, ax=ax)
            ax.set_title(f"{balance_measure} vs {cd_measure}\nr={row['Correlation']:.2f}, p={row['P_Value']:.4f}")
            ax.set_xlabel(balance_measure)
            ax.set_ylabel(cd_measure)

        # Hide any unused subplots
        for j in range(i + 1, 9):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'significant_change_correlations.png'))

    return results


print("Performing change score analysis...")
change_results = perform_change_score_analysis(df)


# ----------------------------------------------------------------------------------
# 6. Mixed-Effects Model
# ----------------------------------------------------------------------------------
def perform_mixed_effects_model(df):
    results = {}

    # Define cognitive dissonance measures as dependent variables
    dependent_vars = [
        'rating_change',
        'rejected_pair_rating_diff',
        'chosen_pair_rating_diff',
        'computer_pair_rating_diff'
    ]

    # Define fixed effects variables (balance metrics)
    itug_metrics = ['duration', 'PC1_std', 'PC1_range', 'PC1_dominant_freq', 'stride_frequency', 'num_strides']
    isway_metrics = ['PC1_std', 'PC1_range', 'PC1_dominant_freq', 'PC1_spectral_entropy', 'sway_area', 'mean_distance',
                     'path_length']

    # Create a wide-to-long format DataFrame for mixed models
    def create_long_data(df, test_type, metrics):
        dfs = []

        for metric in metrics:
            # Pre and post columns for this metric
            pre_col = f'pre_{test_type}_{metric}'
            post_col = f'post_{test_type}_{metric}'

            # Select valid data points
            valid_data = df[['name', pre_col, post_col] + dependent_vars].dropna()

            if len(valid_data) > 5:  # Ensure enough data points
                # Convert to long format
                long_data = pd.DataFrame({
                    'subject': np.repeat(valid_data['name'].values, 2),
                    'time': np.tile(['pre', 'post'], len(valid_data)),
                    f'{test_type}_{metric}': np.concatenate([valid_data[pre_col].values, valid_data[post_col].values])
                })

                # Add cognitive dissonance measures (same for pre and post per subject)
                for dv in dependent_vars:
                    long_data[dv] = np.repeat(valid_data[dv].values, 2)

                dfs.append(long_data)

        # Merge all metrics if any valid data exists
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return None

    # Create long data for ITUG and ISWAY
    long_itug = create_long_data(df, 'itug', itug_metrics)
    long_isway = create_long_data(df, 'isway', isway_metrics)

    # Function to run mixed models
    def run_mixed_models(long_data, test_type, metrics, dependent_vars):
        model_results = {}

        if long_data is None:
            return model_results

        for metric in metrics:
            feature = f'{test_type}_{metric}'

            if feature not in long_data.columns:
                continue

            for dv in dependent_vars:
                try:
                    # Formula: DV ~ feature + time + feature:time + (1|subject)
                    formula = f"{dv} ~ {feature} * time"

                    # Run mixed effects model
                    model = mixedlm(formula, long_data, groups=long_data["subject"])
                    model_fit = model.fit()

                    # Store results
                    model_results[f"{dv}_predicted_by_{feature}"] = {
                        "aic": model_fit.aic,
                        "bic": model_fit.bic,
                        "log_likelihood": model_fit.llf,
                        "parameters": {
                            name: {
                                "estimate": estimate,
                                "std_error": std_err,
                                "p_value": p_value
                            }
                            for name, estimate, std_err, p_value in zip(
                                model_fit.params.index,
                                model_fit.params,
                                model_fit.bse,
                                model_fit.pvalues
                            )
                        }
                    }
                except Exception as e:
                    print(f"Error running mixed model for {feature} predicting {dv}: {e}")

        return model_results

    # Run mixed models for ITUG and ISWAY
    itug_results = run_mixed_models(long_itug, 'itug', itug_metrics, dependent_vars)
    isway_results = run_mixed_models(long_isway, 'isway', isway_metrics, dependent_vars)

    # Combine results
    results = {**itug_results, **isway_results}

    # Save results to file
    with open(os.path.join(output_dir, 'mixed_effects_model.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Create a summary DataFrame
    mixed_model_summary = []

    for model_name, model_results in results.items():
        # Extract dependent variable and predictor
        parts = model_name.split('_predicted_by_')
        dv = parts[0]
        predictor = parts[1]

        # Add model information
        mixed_model_summary.append({
            'Dependent_Variable': dv,
            'Predictor': predictor,
            'AIC': model_results['aic'],
            'BIC': model_results['bic'],
            'Log_Likelihood': model_results['log_likelihood']
        })

    # Convert to DataFrame and save
    mixed_model_df = pd.DataFrame(mixed_model_summary)
    mixed_model_df.to_csv(os.path.join(output_dir, 'mixed_effects_model_summary.csv'), index=False)

    # Create a parameter estimates table
    param_rows = []

    for model_name, model_results in results.items():
        # Extract dependent variable and predictor
        parts = model_name.split('_predicted_by_')
        dv = parts[0]
        predictor = parts[1]

        # Add each parameter
        for param_name, param_info in model_results['parameters'].items():
            param_rows.append({
                'Dependent_Variable': dv,
                'Predictor': predictor,
                'Parameter': param_name,
                'Estimate': param_info['estimate'],
                'Std_Error': param_info['std_error'],
                'P_Value': param_info['p_value'],
                'Significant': param_info['p_value'] < 0.05
            })

    # Convert to DataFrame and save
    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(os.path.join(output_dir, 'mixed_effects_model_parameters.csv'), index=False)

    # Create visualizations
    if len(param_df) > 0:
        # Filter for only the significant interaction effects
        interaction_effects = param_df[
            (param_df['Parameter'].str.contains(':')) &
            (param_df['Significant'])
            ]

        if len(interaction_effects) > 0:
            plt.figure(figsize=(12, len(interaction_effects) * 0.5))

            # Plot significant interaction effects
            plt.barh(
                y=interaction_effects['Parameter'] + ' (' + interaction_effects['Predictor'] + ' → ' +
                  interaction_effects['Dependent_Variable'] + ')',
                width=interaction_effects['Estimate']
            )

            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Estimate')
            plt.title('Significant Interaction Effects in Mixed Models')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mixed_model_interactions.png'))

    return results


print("Performing mixed-effects model analysis...")
mixed_model_results = perform_mixed_effects_model(df)


# ----------------------------------------------------------------------------------
# Integrated Analysis Summary
# ----------------------------------------------------------------------------------
def create_integrated_analysis_summary():
    print("Creating integrated analysis summary...")

    # Create a summary table that combines key findings from all analyses
    summary = {
        "data_overview": {
            "num_participants": len(df),
            "demographic_summary": {
                "age_mean": df['age'].mean(),
                "age_std": df['age'].std(),
                "gender_distribution": df['gender'].value_counts().to_dict(),
                "bmi_mean": df['bmi'].mean(),
                "bmi_std": df['bmi'].std()
            }
        },
        "key_findings": {
            "correlation_analysis": {
                "significant_correlations": []
            },
            "regression_analysis": {
                "significant_models": []
            },
            "anova_results": {
                "significant_changes": []
            },
            "ttest_results": {
                "significant_differences": []
            },
            "change_score_analysis": {
                "significant_correlations": []
            },
            "mixed_effects_models": {
                "significant_interactions": []
            }
        }
    }

    # Add significant correlations from the correlation analysis
    corr_df = pd.read_csv(os.path.join(output_dir, 'repeated_measures_correlation.csv'))
    significant_corrs = corr_df[corr_df['P_Value'] < 0.05].sort_values(by='P_Value')

    for _, row in significant_corrs.iterrows():
        summary["key_findings"]["correlation_analysis"]["significant_correlations"].append({
            "balance_measure": row['Balance_Measure'],
            "cognitive_measure": row['Cognitive_Measure'],
            "correlation": row['Correlation'],
            "p_value": row['P_Value']
        })

    # Add significant regression models
    reg_df = pd.read_csv(os.path.join(output_dir, 'linear_regression_summary.csv'))
    significant_models = reg_df[reg_df['P_Value'] < 0.05].sort_values(by='P_Value')

    for _, row in significant_models.iterrows():
        summary["key_findings"]["regression_analysis"]["significant_models"].append({
            "dependent_variable": row['Dependent_Variable'],
            "predictor_group": row['Predictor_Group'],
            "r_squared": row['R_Squared'],
            "p_value": row['P_Value']
        })

    # Add significant ANOVA results
    anova_df = pd.read_csv(os.path.join(output_dir, 'repeated_measures_anova.csv'))
    significant_anova = anova_df[anova_df['P_Value'] < 0.05].sort_values(by='P_Value')

    for _, row in significant_anova.iterrows():
        summary["key_findings"]["anova_results"]["significant_changes"].append({
            "measure": row['Measure'],
            "f_statistic": row['F_Statistic'],
            "p_value": row['P_Value'],
            "pre_mean": row['Pre_Mean'],
            "post_mean": row['Post_Mean'],
            "change": row['Post_Mean'] - row['Pre_Mean']
        })

    # Add significant t-test results
    ttest_df = pd.read_csv(os.path.join(output_dir, 'paired_ttest.csv'))
    significant_ttests = ttest_df[ttest_df['P_Value'] < 0.05].sort_values(by='P_Value')

    for _, row in significant_ttests.iterrows():
        summary["key_findings"]["ttest_results"]["significant_differences"].append({
            "measure": row['Measure'],
            "t_statistic": row['T_Statistic'],
            "p_value": row['P_Value'],
            "mean_difference": row['Mean_Difference'],
            "cohens_d": row['Cohens_d']
        })

    # Add significant change score correlations
    change_df = pd.read_csv(os.path.join(output_dir, 'change_score_analysis.csv'))
    significant_changes = change_df[change_df['P_Value'] < 0.05].sort_values(by='P_Value')

    for _, row in significant_changes.iterrows():
        summary["key_findings"]["change_score_analysis"]["significant_correlations"].append({
            "balance_change_measure": row['Balance_Change_Measure'],
            "cognitive_measure": row['Cognitive_Measure'],
            "correlation": row['Correlation'],
            "p_value": row['P_Value']
        })

    # Add significant mixed-effects model interactions
    if os.path.exists(os.path.join(output_dir, 'mixed_effects_model_parameters.csv')):
        param_df = pd.read_csv(os.path.join(output_dir, 'mixed_effects_model_parameters.csv'))
        significant_interactions = param_df[
            (param_df['P_Value'] < 0.05) &
            (param_df['Parameter'].str.contains(':'))
            ].sort_values(by='P_Value')

        for _, row in significant_interactions.iterrows():
            summary["key_findings"]["mixed_effects_models"]["significant_interactions"].append({
                "dependent_variable": row['Dependent_Variable'],
                "predictor": row['Predictor'],
                "parameter": row['Parameter'],
                "estimate": row['Estimate'],
                "p_value": row['P_Value']
            })

    # Save summary to file
    with open(os.path.join(output_dir, 'integrated_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    return summary


# Create integrated summary
integrated_summary = create_integrated_analysis_summary()

print("All analyses completed successfully!")
print(f"Results have been saved to: {output_dir}")