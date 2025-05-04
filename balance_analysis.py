import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
import os
from itertools import combinations


# Function to load data
def load_data(file_path):
    """
    Reads JSON format data and converts it to a DataFrame
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If it's data for a single person, handle accordingly
    if isinstance(data, dict):
        data = [data]

    # Convert to DataFrame
    df_list = []
    for person in data:
        person_df = {}

        # Demographic information
        person_df['name'] = person.get('name', '')
        person_df['gender'] = person.get('gender', '')
        person_df['age'] = person.get('age', 0)
        person_df['weight'] = person.get('weight', 0)
        person_df['height'] = person.get('height', 0)
        person_df['bmi'] = person.get('bmi', 0)

        # Pre-ITUG metrics
        pre_itug = person.get('pre_itug_metrics', {})
        for key, value in pre_itug.items():
            person_df[f'pre_itug_{key}'] = value

        # Pre-ISWAY metrics
        pre_isway = person.get('pre_isway_metrics', {})
        for key, value in pre_isway.items():
            person_df[f'pre_isway_{key}'] = value

        # Post-ITUG metrics
        post_itug = person.get('post_itug_metrics', {})
        for key, value in post_itug.items():
            person_df[f'post_itug_{key}'] = value

        # Post-ISWAY metrics
        post_isway = person.get('post_isway_metrics', {})
        for key, value in post_isway.items():
            person_df[f'post_isway_{key}'] = value

        # Cognitive dissonance measurements
        person_df['initial_rating'] = person.get('initial_rating', 0)
        person_df['final_rating'] = person.get('final_rating', 0)
        person_df['rejected_pair_rating_diff'] = person.get('rejected_pair_rating_diff', 0)
        person_df['chosen_pair_rating_diff'] = person.get('chosen_pair_rating_diff', 0)
        person_df['computer_pair_rating_diff'] = person.get('computer_pair_rating_diff', 0)
        person_df['timeout_count'] = person.get('timeout_count', 0)

        # Calculate cognitive dissonance measures
        person_df['average_rating_change'] = abs(person_df['final_rating'] - person_df['initial_rating'])
        person_df['average_rejected_rating_difference'] = abs(person_df['rejected_pair_rating_diff'])

        # Add to the list
        df_list.append(person_df)

    # Combine all person data into a single DataFrame
    df = pd.DataFrame(df_list)
    return df


# Function to calculate balance change scores
def calculate_balance_changes(df):
    """
    Calculate changes in balance parameters between pre and post tests
    """
    balance_changes = pd.DataFrame()
    balance_changes['name'] = df['name']

    # Calculate ITUG changes
    itug_params = ['PC1_std', 'PC1_range', 'PC1_dominant_freq', 'PC1_spectral_entropy', 'stride_frequency']
    for param in itug_params:
        balance_changes[f'itug_{param}_change'] = df[f'post_itug_{param}'] - df[f'pre_itug_{param}']

    # Calculate ISWAY changes
    isway_params = ['PC1_std', 'PC1_range', 'PC1_dominant_freq', 'PC1_spectral_entropy', 'sway_area', 'mean_distance',
                    'path_length']
    for param in isway_params:
        balance_changes[f'isway_{param}_change'] = df[f'post_isway_{param}'] - df[f'pre_isway_{param}']

    # Add cognitive dissonance measures for analysis
    balance_changes['average_rating_change'] = df['average_rating_change']
    balance_changes['average_rejected_rating_difference'] = df['average_rejected_rating_difference']
    balance_changes['chosen_pair_rating_diff'] = df['chosen_pair_rating_diff']
    balance_changes['computer_pair_rating_diff'] = df['computer_pair_rating_diff']

    return balance_changes


# Function for correlation analysis
def correlation_analysis(df, balance_changes):
    """
    Perform correlation analysis between initial balance measures and cognitive dissonance
    """
    print("\n=== CORRELATION ANALYSIS ===\n")

    # Define balance parameters to analyze
    itug_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'stride_frequency']
    isway_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'sway_area', 'mean_distance', 'path_length']

    # Initialize results dictionary
    correlation_results = {
        'test_type': [],
        'parameter': [],
        'correlation_with_rating_change': [],
        'p_value_rating_change': [],
        'correlation_with_rejected_diff': [],
        'p_value_rejected_diff': []
    }

    # Correlations for ITUG measures
    for param in itug_params:
        # Correlation with average rating change
        corr_rating, p_rating = stats.pearsonr(df[f'pre_itug_{param}'], df['average_rating_change'])

        # Correlation with rejected rating difference
        corr_rejected, p_rejected = stats.pearsonr(df[f'pre_itug_{param}'], df['average_rejected_rating_difference'])

        # Store results
        correlation_results['test_type'].append('ITUG')
        correlation_results['parameter'].append(param)
        correlation_results['correlation_with_rating_change'].append(corr_rating)
        correlation_results['p_value_rating_change'].append(p_rating)
        correlation_results['correlation_with_rejected_diff'].append(corr_rejected)
        correlation_results['p_value_rejected_diff'].append(p_rejected)

    # Correlations for ISWAY measures
    for param in isway_params:
        # Correlation with average rating change
        corr_rating, p_rating = stats.pearsonr(df[f'pre_isway_{param}'], df['average_rating_change'])

        # Correlation with rejected rating difference
        corr_rejected, p_rejected = stats.pearsonr(df[f'pre_isway_{param}'], df['average_rejected_rating_difference'])

        # Store results
        correlation_results['test_type'].append('ISWAY')
        correlation_results['parameter'].append(param)
        correlation_results['correlation_with_rating_change'].append(corr_rating)
        correlation_results['p_value_rating_change'].append(p_rating)
        correlation_results['correlation_with_rejected_diff'].append(corr_rejected)
        correlation_results['p_value_rejected_diff'].append(p_rejected)

    # Convert to DataFrame
    corr_df = pd.DataFrame(correlation_results)

    # Add significance indicators
    corr_df['sig_rating_change'] = corr_df['p_value_rating_change'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    corr_df['sig_rejected_diff'] = corr_df['p_value_rejected_diff'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )

    print(corr_df)
    return corr_df


# Function for linear regression analysis
def linear_regression_analysis(df):
    """
    Perform linear regression to predict cognitive dissonance from balance parameters
    """
    print("\n=== LINEAR REGRESSION ANALYSIS ===\n")

    # Define dependent variables (cognitive dissonance measures)
    dependent_vars = ['average_rating_change', 'average_rejected_rating_difference']

    # Define independent variables (balance parameters)
    itug_predictors = ['pre_itug_PC1_std', 'pre_itug_PC1_range', 'pre_itug_PC1_spectral_entropy',
                       'pre_itug_stride_frequency']
    isway_predictors = ['pre_isway_PC1_std', 'pre_isway_PC1_range', 'pre_isway_PC1_spectral_entropy',
                        'pre_isway_sway_area', 'pre_isway_mean_distance', 'pre_isway_path_length']

    regression_results = []

    # Run regressions for each combination of dependent variable and test type
    for dep_var in dependent_vars:
        # ITUG regression
        itug_formula = f"{dep_var} ~ {' + '.join(itug_predictors)}"
        itug_model = ols(itug_formula, data=df).fit()

        print(f"\nRegression for {dep_var} using ITUG parameters:")
        print(itug_model.summary())

        # Store results
        itug_result = {
            'dependent_variable': dep_var,
            'test_type': 'ITUG',
            'r_squared': itug_model.rsquared,
            'adj_r_squared': itug_model.rsquared_adj,
            'f_statistic': itug_model.fvalue,
            'p_value': itug_model.f_pvalue,
            'significant_predictors': [var for var in itug_predictors if itug_model.pvalues[var] < 0.05]
        }
        regression_results.append(itug_result)

        # ISWAY regression
        isway_formula = f"{dep_var} ~ {' + '.join(isway_predictors)}"
        isway_model = ols(isway_formula, data=df).fit()

        print(f"\nRegression for {dep_var} using ISWAY parameters:")
        print(isway_model.summary())

        # Store results
        isway_result = {
            'dependent_variable': dep_var,
            'test_type': 'ISWAY',
            'r_squared': isway_model.rsquared,
            'adj_r_squared': isway_model.rsquared_adj,
            'f_statistic': isway_model.fvalue,
            'p_value': isway_model.f_pvalue,
            'significant_predictors': [var for var in isway_predictors if isway_model.pvalues[var] < 0.05]
        }
        regression_results.append(isway_result)

    # Convert to DataFrame
    reg_results_df = pd.DataFrame(regression_results)
    print("\nRegression Results Summary:")
    print(reg_results_df)

    return reg_results_df


# Function for repeated measures ANOVA
def repeated_measures_anova(df):
    """
    Perform repeated measures ANOVA to compare pre and post balance measures
    """
    print("\n=== REPEATED MEASURES ANOVA ===\n")

    # Prepare data for ANOVA
    # We need to convert from wide to long format

    # Define parameters to analyze
    itug_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'stride_frequency']
    isway_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'sway_area', 'mean_distance', 'path_length']

    anova_results = []

    # ANOVA for ITUG parameters
    for param in itug_params:
        # Prepare data
        anova_data = pd.DataFrame({
            'subject': df['name'],
            'pre': df[f'pre_itug_{param}'],
            'post': df[f'post_itug_{param}']
        })

        # Convert to long format
        anova_data_long = pd.melt(anova_data,
                                  id_vars=['subject'],
                                  value_vars=['pre', 'post'],
                                  var_name='time',
                                  value_name='value')

        try:
            # Run the ANOVA
            aovrm = AnovaRM(anova_data_long, 'value', 'subject', within=['time'])
            res = aovrm.fit()

            result = {
                'test_type': 'ITUG',
                'parameter': param,
                'F_value': res.anova_table['F Value'][0],
                'p_value': res.anova_table['Pr > F'][0],
                'significant': res.anova_table['Pr > F'][0] < 0.05
            }
            anova_results.append(result)

            print(f"\nANOVA results for ITUG {param}:")
            print(res.anova_table)

        except Exception as e:
            print(f"Error running ANOVA for ITUG {param}: {e}")

    # ANOVA for ISWAY parameters
    for param in isway_params:
        # Prepare data
        anova_data = pd.DataFrame({
            'subject': df['name'],
            'pre': df[f'pre_isway_{param}'],
            'post': df[f'post_isway_{param}']
        })

        # Convert to long format
        anova_data_long = pd.melt(anova_data,
                                  id_vars=['subject'],
                                  value_vars=['pre', 'post'],
                                  var_name='time',
                                  value_name='value')

        try:
            # Run the ANOVA
            aovrm = AnovaRM(anova_data_long, 'value', 'subject', within=['time'])
            res = aovrm.fit()

            result = {
                'test_type': 'ISWAY',
                'parameter': param,
                'F_value': res.anova_table['F Value'][0],
                'p_value': res.anova_table['Pr > F'][0],
                'significant': res.anova_table['Pr > F'][0] < 0.05
            }
            anova_results.append(result)

            print(f"\nANOVA results for ISWAY {param}:")
            print(res.anova_table)

        except Exception as e:
            print(f"Error running ANOVA for ISWAY {param}: {e}")

    # Convert to DataFrame
    anova_results_df = pd.DataFrame(anova_results)
    print("\nANOVA Results Summary:")
    print(anova_results_df)

    return anova_results_df


# Function for paired t-tests
def paired_t_tests(df):
    """
    Perform paired t-tests to compare pre and post balance measures
    """
    print("\n=== PAIRED T-TESTS ===\n")

    # Define parameters to analyze
    itug_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'stride_frequency']
    isway_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'sway_area', 'mean_distance', 'path_length']

    t_test_results = []

    # T-tests for ITUG parameters
    for param in itug_params:
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(df[f'pre_itug_{param}'], df[f'post_itug_{param}'])

        # Calculate Cohen's d effect size
        mean_diff = np.mean(df[f'post_itug_{param}'] - df[f'pre_itug_{param}'])
        std_diff = np.std(df[f'post_itug_{param}'] - df[f'pre_itug_{param}'], ddof=1)
        cohen_d = mean_diff / std_diff if std_diff != 0 else 0

        result = {
            'test_type': 'ITUG',
            'parameter': param,
            't_statistic': t_stat,
            'p_value': p_val,
            'mean_difference': mean_diff,
            'effect_size_d': cohen_d,
            'significant': p_val < 0.05
        }
        t_test_results.append(result)

        print(f"\nPaired t-test for ITUG {param}:")
        print(f"t = {t_stat:.4f}, p = {p_val:.4f}, Cohen's d = {cohen_d:.4f}")
        if p_val < 0.05:
            print(
                f"Significant change detected: {'increase' if mean_diff > 0 else 'decrease'} after cognitive dissonance")

    # T-tests for ISWAY parameters
    for param in isway_params:
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(df[f'pre_isway_{param}'], df[f'post_isway_{param}'])

        # Calculate Cohen's d effect size
        mean_diff = np.mean(df[f'post_isway_{param}'] - df[f'pre_isway_{param}'])
        std_diff = np.std(df[f'post_isway_{param}'] - df[f'pre_isway_{param}'], ddof=1)
        cohen_d = mean_diff / std_diff if std_diff != 0 else 0

        result = {
            'test_type': 'ISWAY',
            'parameter': param,
            't_statistic': t_stat,
            'p_value': p_val,
            'mean_difference': mean_diff,
            'effect_size_d': cohen_d,
            'significant': p_val < 0.05
        }
        t_test_results.append(result)

        print(f"\nPaired t-test for ISWAY {param}:")
        print(f"t = {t_stat:.4f}, p = {p_val:.4f}, Cohen's d = {cohen_d:.4f}")
        if p_val < 0.05:
            print(
                f"Significant change detected: {'increase' if mean_diff > 0 else 'decrease'} after cognitive dissonance")

    # Convert to DataFrame
    t_test_results_df = pd.DataFrame(t_test_results)
    print("\nPaired T-test Results Summary:")
    print(t_test_results_df)

    return t_test_results_df


# Function for change score analysis
def change_score_analysis(balance_changes):
    """
    Analyze correlations between cognitive dissonance measures and balance changes
    """
    print("\n=== CHANGE SCORE ANALYSIS ===\n")

    # Define cognitive dissonance measures
    cd_measures = ['average_rating_change', 'average_rejected_rating_difference']

    # Define balance change parameters
    balance_params = [col for col in balance_changes.columns if '_change' in col]

    change_corr_results = []

    # Analyze correlations between cognitive dissonance and balance changes
    for cd_measure in cd_measures:
        for balance_param in balance_params:
            # Skip if the parameter is a cognitive dissonance measure
            if balance_param in cd_measures:
                continue

            # Calculate correlation
            corr, p_val = stats.pearsonr(balance_changes[cd_measure], balance_changes[balance_param])

            result = {
                'cognitive_dissonance_measure': cd_measure,
                'balance_parameter': balance_param,
                'correlation': corr,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            change_corr_results.append(result)

    # Convert to DataFrame
    change_corr_df = pd.DataFrame(change_corr_results)

    # Sort by significance and correlation strength
    change_corr_df = change_corr_df.sort_values(by=['significant', 'correlation'], ascending=[False, False])

    print("\nCorrelations between cognitive dissonance and balance changes:")
    print(change_corr_df)

    # Create visualizations for significant correlations
    significant_corrs = change_corr_df[change_corr_df['significant']]

    if len(significant_corrs) > 0:
        print("\nGenerating scatter plots for significant correlations...")

        for _, row in significant_corrs.iterrows():
            cd_measure = row['cognitive_dissonance_measure']
            balance_param = row['balance_parameter']
            corr = row['correlation']
            p_val = row['p_value']

            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=balance_changes[cd_measure], y=balance_changes[balance_param])
            sns.regplot(x=balance_changes[cd_measure], y=balance_changes[balance_param], scatter=False)

            plt.title(f"Correlation between {cd_measure} and {balance_param}\nr = {corr:.3f}, p = {p_val:.3f}")
            plt.xlabel(cd_measure)
            plt.ylabel(balance_param)

            # Save the plot
            plt.savefig(f"correlation_{cd_measure}_{balance_param}.png")

    return change_corr_df


# Function for mixed effects model
def mixed_effects_model(df):
    """
    Run mixed effects models to analyze both the impact of initial balance on cognitive dissonance
    and the effect of cognitive dissonance on subsequent balance
    """
    print("\n=== MIXED EFFECTS MODEL ANALYSIS ===\n")

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
            pre_data[param] = row[f'pre_itug_{param}']

        itug_data.append(pre_data)

        # Post-test data
        post_data = {
            'subject': subject,
            'time': 'post',
            'cognitive_dissonance': row['average_rating_change'],
            'rejected_rating_diff': row['average_rejected_rating_difference']
        }

        # Add all ITUG parameters
        for param in itug_params:
            post_data[param] = row[f'post_itug_{param}']

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
            pre_data[param] = row[f'pre_isway_{param}']

        isway_data.append(pre_data)

        # Post-test data
        post_data = {
            'subject': subject,
            'time': 'post',
            'cognitive_dissonance': row['average_rating_change'],
            'rejected_rating_diff': row['average_rejected_rating_difference']
        }

        # Add all ISWAY parameters
        for param in isway_params:
            post_data[param] = row[f'post_isway_{param}']

        isway_data.append(post_data)

    # Convert to DataFrame
    isway_long = pd.DataFrame(isway_data)

    mixed_model_results = []

    # Run mixed effects models for ITUG parameters
    print("\nMixed Effects Models for ITUG parameters:")
    for param in itug_params:
        try:
            # Filter only post data to analyze effect of pre-values on cognitive dissonance
            post_data = itug_long[itug_long['time'] == 'post'].copy()

            # Add pre-values to the post data
            for idx, subj in enumerate(post_data['subject']):
                pre_row = itug_long[(itug_long['subject'] == subj) & (itug_long['time'] == 'pre')]
                post_data.loc[post_data.index[idx], f'pre_{param}'] = pre_row[param].values[0]

            # Model 1: Effect of pre-balance on cognitive dissonance
            formula1 = f"cognitive_dissonance ~ pre_{param}"
            model1 = smf.ols(formula1, data=post_data).fit()

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

            # Model 2: Effect of cognitive dissonance on post-balance, controlling for pre-balance
            formula2 = f"{param} ~ cognitive_dissonance + pre_{param}"
            model2 = smf.ols(formula2, data=post_data).fit()

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
                post_data.loc[post_data.index[idx], f'pre_{param}'] = pre_row[param].values[0]

            # Model 1: Effect of pre-balance on cognitive dissonance
            formula1 = f"cognitive_dissonance ~ pre_{param}"
            model1 = smf.ols(formula1, data=post_data).fit()

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

            # Model 2: Effect of cognitive dissonance on post-balance, controlling for pre-balance
            formula2 = f"{param} ~ cognitive_dissonance + pre_{param}"
            model2 = smf.ols(formula2, data=post_data).fit()

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
            print(f"Error in mixed effects model for ISWAY {param}: {e}")

    # Convert to DataFrame
    mixed_model_df = pd.DataFrame(mixed_model_results)
    print("\nMixed Effects Model Results Summary:")
    print(mixed_model_df)

    return mixed_model_df


# Function to run comprehensive analysis
def run_analysis(file_path):
    """