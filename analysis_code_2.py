import os
import glob
import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4

# Define balance parameters - keeping the original parameters from your code
itug_params = ['PC1_std', 'PC1_range', 'PC1_mean', 'PC2_std', 'PC2_range', 'PC2_mean']
isway_params = ['PC1_std', 'PC1_range', 'PC1_mean', 'mean_distance', 'sway_area']


def load_data(base_dir):
    """
    Load participant data from participant_summary.json files.

    Args:
        base_dir: The base directory containing the sensor_analysis folder

    Returns:
        DataFrame: Containing all participant data
    """
    sensor_dir = os.path.join(base_dir, "sensor_analysis")
    df_list = []

    # Find all participant directories (format: Name-Gender-Age-Weight-Height)
    participant_dirs = glob.glob(os.path.join(sensor_dir, "*-[EK]-*-*-*"))

    print(f"Found {len(participant_dirs)} participant directories")

    for participant_dir in participant_dirs:
        participant_name = os.path.basename(participant_dir)
        participant_summary_path = os.path.join(participant_dir, 'participant_summary.json')

        if not os.path.exists(participant_summary_path):
            print(f"Warning: Missing participant summary file for {participant_name}. Skipping.")
            continue

        try:
            with open(participant_summary_path, 'r', encoding='utf-8') as f:
                participant_data = json.load(f)

            # Extract demographic data
            person_df = pd.DataFrame({
                'name': [participant_data['name']],
                'gender': [participant_data['gender']],
                'age': [participant_data['age']],
                'weight': [participant_data['weight']],
                'height': [participant_data['height']],
                'bmi': [participant_data['bmi']]
            })

            # Process pre-ITUG metrics
            pre_itug = participant_data.get('pre_itug_metrics', {})
            if pre_itug and pre_itug.get('valid', False):
                for param in itug_params:
                    # Handle PC2 parameters which might not be in the original data
                    if param.startswith('PC2') and param not in pre_itug:
                        person_df[f'pre_itug_{param}'] = 0
                        print(f"Warning: Missing {param} in pre_itug for {participant_name}")
                    else:
                        # For PC1 parameters and others
                        param_key = param
                        person_df[f'pre_itug_{param}'] = pre_itug.get(param_key, 0)
            else:
                print(f"Warning: Invalid or missing pre_itug_metrics for {participant_name}")
                continue  # Skip this participant if pre_itug data is invalid

            # Process post-ITUG metrics
            post_itug = participant_data.get('post_itug_metrics', {})
            if post_itug and post_itug.get('valid', False):
                for param in itug_params:
                    # Handle PC2 parameters which might not be in the original data
                    if param.startswith('PC2') and param not in post_itug:
                        person_df[f'post_itug_{param}'] = 0
                        print(f"Warning: Missing {param} in post_itug for {participant_name}")
                    else:
                        # For PC1 parameters and others
                        param_key = param
                        person_df[f'post_itug_{param}'] = post_itug.get(param_key, 0)
            else:
                print(f"Warning: Invalid or missing post_itug_metrics for {participant_name}")
                continue  # Skip this participant if post_itug data is invalid

            # Process pre-ISWAY metrics
            pre_isway = participant_data.get('pre_isway_metrics', {})
            if pre_isway and pre_isway.get('valid', False):
                for param in isway_params:
                    person_df[f'pre_isway_{param}'] = pre_isway.get(param, 0)
            else:
                print(f"Warning: Invalid or missing pre_isway_metrics for {participant_name}")
                continue  # Skip this participant if pre_isway data is invalid

            # Process post-ISWAY metrics
            post_isway = participant_data.get('post_isway_metrics', {})
            if post_isway and post_isway.get('valid', False):
                for param in isway_params:
                    person_df[f'post_isway_{param}'] = post_isway.get(param, 0)
            else:
                print(f"Warning: Invalid or missing post_isway_metrics for {participant_name}")
                continue  # Skip this participant if post_isway data is invalid

            # Process food choice ratings
            person_df['average_initial_rating'] = participant_data.get('initial_rating', 0)
            person_df['average_final_rating'] = participant_data.get('final_rating', 0)
            person_df['average_rejected_difference'] = participant_data.get('rejected_pair_rating_diff', 0)
            person_df['average_rating_change'] = abs(
                person_df['average_final_rating'] - person_df['average_initial_rating'])

            # Additional food choice metrics
            person_df['chosen_pair_rating_diff'] = participant_data.get('chosen_pair_rating_diff', 0)
            person_df['computer_pair_rating_diff'] = participant_data.get('computer_pair_rating_diff', 0)
            person_df['timeout_count'] = participant_data.get('timeout_count', 0)

            # Debug print
            print(f"Successfully processed participant {participant_name}")
            print(f"  average_initial_rating: {person_df['average_initial_rating'].iloc[0]}")
            print(f"  average_final_rating: {person_df['average_final_rating'].iloc[0]}")
            print(f"  average_rating_change: {person_df['average_rating_change'].iloc[0]}")
            print(f"  average_rejected_difference: {person_df['average_rejected_difference'].iloc[0]}")

            df_list.append(person_df)

        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in participant summary file: {participant_summary_path}. Skipping.")
        except Exception as e:
            print(f"Error processing {participant_name}: {str(e)}")

    if not df_list:
        raise ValueError("No valid participants found. Analysis cannot proceed.")

    df = pd.concat(df_list, ignore_index=True)
    print(f"Total participants included in analysis: {len(df)}")
    return df


def calculate_balance_changes(df):
    """Calculate changes in balance parameters from pre to post test."""
    balance_changes = pd.DataFrame()
    balance_changes['name'] = df['name']

    # Add demographic information
    balance_changes['gender'] = df['gender']
    balance_changes['age'] = df['age']
    balance_changes['bmi'] = df['bmi']

    for param in itug_params:
        balance_changes[f'itug_{param}_change'] = df[f'post_itug_{param}'] - df[f'pre_itug_{param}']
    for param in isway_params:
        balance_changes[f'isway_{param}_change'] = df[f'post_isway_{param}'] - df[f'pre_isway_{param}']

    # Add cognitive dissonance measures
    balance_changes['average_rating_change'] = df['average_rating_change']
    balance_changes['average_rejected_difference'] = df['average_rejected_difference']
    balance_changes['chosen_pair_rating_diff'] = df['chosen_pair_rating_diff']
    balance_changes['computer_pair_rating_diff'] = df['computer_pair_rating_diff']

    return balance_changes


def correlation_analysis(df, balance_changes):
    """Perform correlation analysis between balance parameters and food choice metrics."""
    results = []
    print("\n=== CORRELATION ANALYSIS ===\n")

    # Check for constant variables
    if df['average_rating_change'].nunique() <= 1:
        print("Warning: average_rating_change is constant. Skipping correlations with this variable.")
    if df['average_rejected_difference'].nunique() <= 1:
        print("Warning: average_rejected_difference is constant. Skipping correlations with this variable.")

    # Add chosen_pair_rating_diff to the analysis
    cd_measures = ['average_rating_change', 'average_rejected_difference', 'chosen_pair_rating_diff']

    for param in itug_params:
        for cd_measure in cd_measures:
            if df[cd_measure].nunique() > 1:
                x = df[f'pre_itug_{param}']
                y = df[cd_measure]
                if x.var() > 0 and y.var() > 0:
                    corr, p = stats.pearsonr(x, y)
                    significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    results.append({
                        'test_type': 'ITUG',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'correlation': corr,
                        'p_value': p,
                        'significance': significance
                    })
                else:
                    results.append({
                        'test_type': 'ITUG',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'correlation': np.nan,
                        'p_value': np.nan,
                        'significance': 'ns'
                    })

    for param in isway_params:
        for cd_measure in cd_measures:
            if df[cd_measure].nunique() > 1:
                x = df[f'pre_isway_{param}']
                y = df[cd_measure]
                if x.var() > 0 and y.var() > 0:
                    corr, p = stats.pearsonr(x, y)
                    significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    results.append({
                        'test_type': 'ISWAY',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'correlation': corr,
                        'p_value': p,
                        'significance': significance
                    })
                else:
                    results.append({
                        'test_type': 'ISWAY',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'correlation': np.nan,
                        'p_value': np.nan,
                        'significance': 'ns'
                    })

    return pd.DataFrame(results)


def linear_regression_analysis(df):
    """Perform linear regression analysis."""
    results = []
    print("\n=== LINEAR REGRESSION ANALYSIS ===\n")

    cd_measures = ['average_rating_change', 'average_rejected_difference', 'chosen_pair_rating_diff']

    for param in itug_params + isway_params:
        for cd_measure in cd_measures:
            if df[cd_measure].nunique() > 1:
                y = df[cd_measure]
                prefix = 'pre_itug_' if param in itug_params else 'pre_isway_'
                X = df[[prefix + param, 'age', 'bmi']]
                X = sm.add_constant(X)
                if y.var() > 0:
                    model = sm.OLS(y, X).fit()
                    results.append({
                        'test_type': 'ITUG' if param in itug_params else 'ISWAY',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'r_squared': model.rsquared,
                        'f_statistic': model.fvalue,
                        'p_value': model.f_pvalue,
                        'significant': model.f_pvalue < 0.05
                    })
                else:
                    results.append({
                        'test_type': 'ITUG' if param in itug_params else 'ISWAY',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'r_squared': np.nan,
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    })

    return pd.DataFrame(results)


def anova_analysis(df):
    """Perform ANOVA analysis to test differences between pre and post measures."""
    results = []
    print("\n=== ANOVA ANALYSIS ===\n")

    for param in itug_params:
        data = pd.melt(df, id_vars=['name'], value_vars=[f'pre_itug_{param}', f'post_itug_{param}'],
                       var_name='time', value_name='value')
        model = ols('value ~ time', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        results.append({
            'test_type': 'ITUG',
            'parameter': param,
            'F_value': anova_table.loc['time', 'F'],
            'p_value': anova_table.loc['time', 'PR(>F)'],
            'significant': anova_table.loc['time', 'PR(>F)'] < 0.05
        })

    for param in isway_params:
        data = pd.melt(df, id_vars=['name'], value_vars=[f'pre_isway_{param}', f'post_isway_{param}'],
                       var_name='time', value_name='value')
        model = ols('value ~ time', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        results.append({
            'test_type': 'ISWAY',
            'parameter': param,
            'F_value': anova_table.loc['time', 'F'],
            'p_value': anova_table.loc['time', 'PR(>F)'],
            'significant': anova_table.loc['time', 'PR(>F)'] < 0.05
        })

    return pd.DataFrame(results)


def paired_ttest_analysis(df):
    """Perform paired t-test analysis between pre and post measures."""
    results = []
    print("\n=== PAIRED T-TEST ANALYSIS ===\n")

    for param in itug_params:
        pre = df[f'pre_itug_{param}']
        post = df[f'post_itug_{param}']
        if pre.var() > 0 and post.var() > 0:
            t_stat, p_val = stats.ttest_rel(pre, post)
            results.append({
                'test_type': 'ITUG',
                'parameter': param,
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
        else:
            results.append({
                'test_type': 'ITUG',
                'parameter': param,
                't_statistic': np.nan,
                'p_value': np.nan,
                'significant': False
            })

    for param in isway_params:
        pre = df[f'pre_isway_{param}']
        post = df[f'post_isway_{param}']
        if pre.var() > 0 and post.var() > 0:
            t_stat, p_val = stats.ttest_rel(pre, post)
            results.append({
                'test_type': 'ISWAY',
                'parameter': param,
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
        else:
            results.append({
                'test_type': 'ISWAY',
                'parameter': param,
                't_statistic': np.nan,
                'p_value': np.nan,
                'significant': False
            })

    return pd.DataFrame(results)


def change_score_analysis(balance_changes, df):
    """Analyze correlations between balance changes and food choice metrics."""
    results = []
    print("\n=== CHANGE SCORE ANALYSIS ===\n")

    cd_measures = ['average_rating_change', 'average_rejected_difference', 'chosen_pair_rating_diff']

    for param in itug_params:
        for cd_measure in cd_measures:
            if df[cd_measure].nunique() > 1:
                x = balance_changes[f'itug_{param}_change']
                y = df[cd_measure]
                if x.var() > 0 and y.var() > 0:
                    corr, p = stats.pearsonr(x, y)
                    significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    results.append({
                        'test_type': 'ITUG',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'correlation': corr,
                        'p_value': p,
                        'significance': significance
                    })
                else:
                    results.append({
                        'test_type': 'ITUG',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'correlation': np.nan,
                        'p_value': np.nan,
                        'significance': 'ns'
                    })

    for param in isway_params:
        for cd_measure in cd_measures:
            if df[cd_measure].nunique() > 1:
                x = balance_changes[f'isway_{param}_change']
                y = df[cd_measure]
                if x.var() > 0 and y.var() > 0:
                    corr, p = stats.pearsonr(x, y)
                    significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    results.append({
                        'test_type': 'ISWAY',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'correlation': corr,
                        'p_value': p,
                        'significance': significance
                    })
                else:
                    results.append({
                        'test_type': 'ISWAY',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'correlation': np.nan,
                        'p_value': np.nan,
                        'significance': 'ns'
                    })

    return pd.DataFrame(results)


def mixed_effects_model(df):
    """Perform mixed effects modeling."""
    results = []
    print("\n=== MIXED EFFECTS MODEL ===\n")

    cd_measures = ['average_rating_change', 'average_rejected_difference', 'chosen_pair_rating_diff']
    mixed_model_results = []

    # Prepare data for mixed effects model
    # We need to restructure the data to have both pre and post measurements in a single dataframe

    # First, create a dataframe for ITUG measurements
    itug_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'stride_frequency']
    itug_data = []

    for _, row in df.iterrows():
        subject = row['name']

        # Pre-test data
        pre_data = {
            'subject': subject,
            'time': 'pre',
            'cognitive_dissonance': None  # Not experienced yet
        }

        # Add all ITUG parameters
        for param in itug_params:
            pre_data[param] = row.get(f'pre_itug_{param}', None)

        itug_data.append(pre_data)

        # Post-test data
        post_data = {
            'subject': subject,
            'time': 'post',
            'cognitive_dissonance': row.get('average_rating_change', None),
            'rejected_rating_diff': row.get('average_rejected_difference', None)
        }

        # Add all ITUG parameters
        for param in itug_params:
            post_data[param] = row.get(f'post_itug_{param}', None)

        itug_data.append(post_data)

    # Convert to DataFrame
    itug_long = pd.DataFrame(itug_data)

    # Same for ISWAY
    isway_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'sway_area', 'mean_distance', 'path_length']
    isway_data = []

    for _, row in df.iterrows():
        subject = row['name']

        # Pre-test data
        pre_data = {
            'subject': subject,
            'time': 'pre',
            'cognitive_dissonance': None  # Not experienced yet
        }

        # Add all ISWAY parameters
        for param in isway_params:
            pre_data[param] = row.get(f'pre_isway_{param}', None)

        isway_data.append(pre_data)

        # Post-test data
        post_data = {
            'subject': subject,
            'time': 'post',
            'cognitive_dissonance': row.get('average_rating_change', None),
            'rejected_rating_diff': row.get('average_rejected_difference', None)
        }

        # Add all ISWAY parameters
        for param in isway_params:
            post_data[param] = row.get(f'post_isway_{param}', None)

        isway_data.append(post_data)

    # Convert to DataFrame
    isway_long = pd.DataFrame(isway_data)

    # Run mixed effects models for ITUG parameters
    print("\nMixed Effects Models for ITUG parameters:")
    for param in itug_params:
        try:
            # Filter only post data to analyze effect of pre-values on cognitive dissonance
            post_data = itug_long[itug_long['time'] == 'post'].copy()

            # Add pre-values to the post data
            for idx, subj in enumerate(post_data['subject']):
                pre_row = itug_long[(itug_long['subject'] == subj) & (itug_long['time'] == 'pre')]
                if not pre_row.empty:
                    post_data.loc[post_data.index[idx], f'pre_{param}'] = pre_row[param].values[0]

            # Model 1: Effect of pre-balance on cognitive dissonance
            formula1 = f"cognitive_dissonance ~ pre_{param}"
            try:
                model1 = sm.OLS(post_data['cognitive_dissonance'],
                                sm.add_constant(post_data[f'pre_{param}'])).fit()

                print(f"\nEffect of pre-{param} on cognitive dissonance:")
                print(model1.summary())

                result1 = {
                    'test_type': 'ITUG',
                    'parameter': param,
                    'model_type': 'pre_balance_on_cd',
                    'coefficient': model1.params[f'pre_{param}'],
                    'p_value': model1.pvalues[f'pre_{param}'],
                    'r_squared': model1.rsquared,
                    'significant': model1.pvalues[f'pre_{param}'] < 0.05
                }
                mixed_model_results.append(result1)
            except Exception as e:
                print(f"Error in model 1 for ITUG {param}: {e}")

            # Model 2: Effect of cognitive dissonance on post-balance, controlling for pre-balance
            formula2 = f"{param} ~ cognitive_dissonance + pre_{param}"
            try:
                X = sm.add_constant(post_data[['cognitive_dissonance', f'pre_{param}']])
                model2 = sm.OLS(post_data[param], X).fit()

                print(f"\nEffect of cognitive dissonance on post-{param} (controlling for pre-{param}):")
                print(model2.summary())

                result2 = {
                    'test_type': 'ITUG',
                    'parameter': param,
                    'model_type': 'cd_on_post_balance',
                    'coefficient': model2.params['cognitive_dissonance'],
                    'p_value': model2.pvalues['cognitive_dissonance'],
                    'r_squared': model2.rsquared,
                    'significant': model2.pvalues['cognitive_dissonance'] < 0.05
                }
                mixed_model_results.append(result2)
            except Exception as e:
                print(f"Error in model 2 for ITUG {param}: {e}")
        except Exception as e:
            print(f"Error in mixed effects model for ITUG {param}: {e}")

    # Run mixed effects models for ISWAY parameters
    print("\nMixed Effects Models for ISWAY parameters:")
    for param in isway_params:
        try:
            # Filter only post data to analyze effect of pre-values on cognitive dissonance
            post_data = isway_long[isway_long['time'] == 'post'].copy()

            # Add pre-values to the post data
            for idx, subj in enumerate(post_data['subject']):
                pre_row = isway_long[(isway_long['subject'] == subj) & (isway_long['time'] == 'pre')]
                if not pre_row.empty:
                    post_data.loc[post_data.index[idx], f'pre_{param}'] = pre_row[param].values[0]

            # Model 1: Effect of pre-balance on cognitive dissonance
            formula1 = f"cognitive_dissonance ~ pre_{param}"
            try:
                model1 = sm.OLS(post_data['cognitive_dissonance'],
                                sm.add_constant(post_data[f'pre_{param}'])).fit()

                print(f"\nEffect of pre-{param} on cognitive dissonance:")
                print(model1.summary())

                result1 = {
                    'test_type': 'ISWAY',
                    'parameter': param,
                    'model_type': 'pre_balance_on_cd',
                    'coefficient': model1.params[f'pre_{param}'],
                    'p_value': model1.pvalues[f'pre_{param}'],
                    'r_squared': model1.rsquared,
                    'significant': model1.pvalues[f'pre_{param}'] < 0.05
                }
                mixed_model_results.append(result1)
            except Exception as e:
                print(f"Error in model 1 for ISWAY {param}: {e}")

            # Model 2: Effect of cognitive dissonance on post-balance, controlling for pre-balance
            formula2 = f"{param} ~ cognitive_dissonance + pre_{param}"
            try:
                X = sm.add_constant(post_data[['cognitive_dissonance', f'pre_{param}']])
                model2 = sm.OLS(post_data[param], X).fit()

                print(f"\nEffect of cognitive dissonance on post-{param} (controlling for pre-{param}):")
                print(model2.summary())

                result2 = {
                    'test_type': 'ISWAY',
                    'parameter': param,
                    'model_type': 'cd_on_post_balance',
                    'coefficient': model2.params['cognitive_dissonance'],
                    'p_value': model2.pvalues['cognitive_dissonance'],
                    'r_squared': model2.rsquared,
                    'significant': model2.pvalues['cognitive_dissonance'] < 0.05
                }
                mixed_model_results.append(result2)
            except Exception as e:
                print(f"Error in model 2 for ISWAY {param}: {e}")

            # Additional model for rejected rating difference
            formula3 = f"{param} ~ rejected_rating_diff + pre_{param}"
            try:
                X = sm.add_constant(post_data[['rejected_rating_diff', f'pre_{param}']])
                model3 = sm.OLS(post_data[param], X).fit()

                print(f"\nEffect of rejected rating difference on post-{param} (controlling for pre-{param}):")
                print(model3.summary())

                result3 = {
                    'test_type': 'ISWAY',
                    'parameter': param,
                    'model_type': 'rejected_diff_on_post_balance',
                    'coefficient': model3.params['rejected_rating_diff'],
                    'p_value': model3.pvalues['rejected_rating_diff'],
                    'r_squared': model3.rsquared,
                    'significant': model3.pvalues['rejected_rating_diff'] < 0.05
                }
                mixed_model_results.append(result3)
            except Exception as e:
                print(f"Error in model 3 for ISWAY {param}: {e}")
        except Exception as e:
            print(f"Error in mixed effects model for ISWAY {param}: {e}")

    return pd.DataFrame(mixed_model_results)


def demographic_summary(df):
    """Create a summary of demographic information."""
    demo_summary = {
        'total_participants': len(df),
        'gender_distribution': df['gender'].value_counts().to_dict(),
        'age_mean': df['age'].mean(),
        'age_std': df['age'].std(),
        'age_range': [df['age'].min(), df['age'].max()],
        'bmi_mean': df['bmi'].mean(),
        'bmi_std': df['bmi'].std(),
        'bmi_range': [df['bmi'].min(), df['bmi'].max()]
    }

    print("\n=== DEMOGRAPHIC SUMMARY ===\n")
    print(f"Total participants: {demo_summary['total_participants']}")
    print(f"Gender distribution: {demo_summary['gender_distribution']}")
    print(
        f"Age: Mean={demo_summary['age_mean']:.2f}, SD={demo_summary['age_std']:.2f}, Range={demo_summary['age_range']}")
    print(
        f"BMI: Mean={demo_summary['bmi_mean']:.2f}, SD={demo_summary['bmi_std']:.2f}, Range={demo_summary['bmi_range']}")

    return demo_summary


def create_visualizations(results, df, output_dir):
    """Create and save visualizations for the analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Save participant data for reference
    df.to_csv(os.path.join(output_dir, 'participant_data.csv'), index=False)

    # Correlation Heatmap
    corr_df = results['correlations']
    if not corr_df.empty and not corr_df['correlation'].isna().all():
        for cd_measure in ['average_rating_change', 'average_rejected_difference', 'chosen_pair_rating_diff']:
            measure_df = corr_df[corr_df['cd_measure'] == cd_measure]
            if not measure_df.empty:
                pivot_data = measure_df.pivot_table(
                    index=['test_type', 'parameter'],
                    values='correlation'
                )
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_data, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
                plt.title(f'Correlation Heatmap - {cd_measure}')
                plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{cd_measure}.png'))
                plt.close()
    else:
        print("Skipping correlation heatmap: all correlations are NaN.")

    # Paired T-Test Bar Plots
    ttest_df = results['ttest']
    for test_type in ['ITUG', 'ISWAY']:
        df_subset = ttest_df[ttest_df['test_type'] == test_type]
        if not df_subset.empty:
            plt.figure(figsize=(10, 6))
            bars = sns.barplot(x='parameter', y='t_statistic', hue='significant', data=df_subset)
            plt.title(f'{test_type} Paired T-Test Results')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Add p-values above bars
            for i, p in enumerate(df_subset['p_value']):
                if not np.isnan(p):
                    bars.annotate(f'p={p:.3f}',
                                  (i, df_subset['t_statistic'].iloc[i]),
                                  ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{test_type}_ttest_barplot.png'))
            plt.close()

    # Pre vs Post Comparison Plots
    for test_type, params in [('ITUG', itug_params), ('ISWAY', isway_params)]:
        for param in params:
            plt.figure(figsize=(10, 6))
            pre_column = f'pre_{test_type.lower()}_{param}'
            post_column = f'post_{test_type.lower()}_{param}'

            # Create a dataframe for plotting
            plot_df = pd.DataFrame({
                'Pre': df[pre_column],
                'Post': df[post_column]
            })

            # Plot means with error bars
            means = plot_df.mean()
            sems = plot_df.sem()

            plt.bar(['Pre', 'Post'], means, yerr=sems, capsize=10)
            plt.title(f'{test_type} {param} - Pre vs Post Comparison')
            plt.ylabel('Value')

            # Add p-value from t-test
            t_result = ttest_df[(ttest_df['test_type'] == test_type) & (ttest_df['parameter'] == param)]
            if not t_result.empty and not np.isnan(t_result['p_value'].iloc[0]):
                p_val = t_result['p_value'].iloc[0]
                sig_text = f'p={p_val:.3f}'
                if p_val < 0.05:
                    sig_text += ' *'
                if p_val < 0.01:
                    sig_text += '*'
                if p_val < 0.001:
                    sig_text += '*'

                plt.text(0.5, max(means) * 1.1, sig_text, ha='center')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{test_type}_{param}_pre_post_comparison.png'))
            plt.close()

    # Mixed Effects Model Forest Plot
    mixed_df = results['mixed_effects']
    if not mixed_df.empty and not mixed_df['coefficient'].isna().all():
        for model_type in mixed_df['model_type'].unique():
            measure_df = mixed_df[mixed_df['model_type'] == model_type]
            if not measure_df.empty:
                plt.figure(figsize=(10, 8))
                colors = ['green' if p < 0.05 else 'gray' for p in measure_df['p_value']]
                plt.errorbar(
                    x=measure_df['coefficient'],
                    y=range(len(measure_df)),
                    fmt='o',
                    markersize=8,
                    color=colors
                )
                plt.yticks(range(len(measure_df)),
                           [f"{row['test_type']} {row['parameter']}" for _, row in measure_df.iterrows()])
                plt.axvline(0, color='black', linestyle='--')

                model_type_labels = {
                    'pre_balance_on_cd': 'Effect of Initial Balance on Cognitive Dissonance',
                    'cd_on_post_balance': 'Effect of Cognitive Dissonance on Final Balance',
                    'rejected_diff_on_post_balance': 'Effect of Rejected Rating Difference on Final Balance'
                }

                plt.title(model_type_labels.get(model_type, model_type))
                plt.xlabel('Coefficient')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'mixed_effects_{model_type}_plot.png'))
                plt.close()
    else:
        print("Skipping mixed effects forest plot: all coefficients are NaN.")

    # Gender comparison for balance parameters
    for test_type, params in [('ITUG', itug_params), ('ISWAY', isway_params)]:
        for param in params:
            pre_column = f'pre_{test_type.lower()}_{param}'
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='gender', y=pre_column, data=df)
            plt.title(f'{test_type} {param} by Gender')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{test_type}_{param}_gender_comparison.png'))
            plt.close()

    # Age vs Balance Parameters
    for test_type, params in [('ITUG', itug_params), ('ISWAY', isway_params)]:
        for param in params:
            pre_column = f'pre_{test_type.lower()}_{param}'
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='age', y=pre_column, hue='gender', data=df)
            plt.title(f'Age vs {test_type} {param}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'age_vs_{test_type}_{param}.png'))
            plt.close()

    # Rating change histograms
    plt.figure(figsize=(10, 6))
    sns.histplot(df['average_rating_change'], kde=True)
    plt.title('Distribution of Rating Changes')
    plt.xlabel('Absolute Rating Change')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_change_distribution.png'))
    plt.close()

    # Correlation between balance changes and rating changes
    change_score_df = results['change_score']
    if not change_score_df['correlation'].isna().all():
        for cd_measure in ['average_rating_change', 'average_rejected_difference', 'chosen_pair_rating_diff']:
            measure_df = change_score_df[change_score_df['cd_measure'] == cd_measure]
            if not measure_df.empty:
                plt.figure(figsize=(10, 8))
                pivot_data = measure_df.pivot_table(
                    index=['test_type', 'parameter'],
                    values='correlation'
                )
                sns.heatmap(pivot_data, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
                plt.title(f'Balance Change to {cd_measure} Correlation')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'change_correlation_{cd_measure}.png'))
                plt.close()

    # Summary statistics for all variables
    summary_stats = df.describe()
    summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))


def main(base_dir, output_dir):
    """
    Main function to run all analyses.

    Args:
        base_dir: Base directory containing sensor_analysis folder
        output_dir: Directory to save analysis results
    """
    print(f"Loading data from {base_dir}")
    df = load_data(base_dir)

    # Generate demographic summary
    demo_summary = demographic_summary(df)

    # Calculate balance changes
    balance_changes = calculate_balance_changes(df)

    # Run all analyses
    correlations = correlation_analysis(df, balance_changes)
    regression = linear_regression_analysis(df)
    anova = anova_analysis(df)
    ttest = paired_ttest_analysis(df)
    change_score = change_score_analysis(balance_changes, df)
    mixed_effects = mixed_effects_model(df)

    # Combine results
    results = {
        'demographics': demo_summary,
        'correlations': correlations,
        'regression': regression,
        'anova': anova,
        'ttest': ttest,
        'change_score': change_score,
        'mixed_effects': mixed_effects
    }

    # Create visualizations
    create_visualizations(results, df, output_dir)

    # Save all results to CSV
    os.makedirs(output_dir, exist_ok=True)
    correlations.to_csv(os.path.join(output_dir, 'correlation_results.csv'), index=False)
    regression.to_csv(os.path.join(output_dir, 'regression_results.csv'), index=False)
    anova.to_csv(os.path.join(output_dir, 'anova_results.csv'), index=False)
    ttest.to_csv(os.path.join(output_dir, 'ttest_results.csv'), index=False)
    change_score.to_csv(os.path.join(output_dir, 'change_score_results.csv'), index=False)
    mixed_effects.to_csv(os.path.join(output_dir, 'mixed_effects_results.csv'), index=False)

    # Also save as JSON
    with open(os.path.join(output_dir, 'demographic_summary.json'), 'w') as f:
        json.dump(demo_summary, f, indent=2)

    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze sensor and food rating data')
    parser.add_argument('--base_dir', default='analysis_results', help='Base directory with analysis results')
    parser.add_argument('--output_dir', default='analysis_results/advanced_analysis',
                        help='Output directory for analysis')

    args = parser.parse_args()

    main(args.base_dir, args.output_dir)