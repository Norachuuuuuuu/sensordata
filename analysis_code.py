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
    for param in itug_params:
        balance_changes[f'itug_{param}_change'] = df[f'post_itug_{param}'] - df[f'pre_itug_{param}']
    for param in isway_params:
        balance_changes[f'isway_{param}_change'] = df[f'post_isway_{param}'] - df[f'pre_isway_{param}']
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

    for param in itug_params + isway_params:
        for cd_measure in cd_measures:
            if df[cd_measure].nunique() > 1:
                prefix = 'pre_itug_' if param in itug_params else 'pre_isway_'
                data = df[[cd_measure, prefix + param, 'age', 'bmi', 'name']].copy()
                data = data.rename(columns={cd_measure: 'cd_measure', prefix + param: 'balance_param'})
                if data['cd_measure'].var() > 0 and data['balance_param'].var() > 0:
                    try:
                        model = mixedlm("cd_measure ~ balance_param + age + bmi", data, groups=data['name']).fit()
                        results.append({
                            'test_type': 'ITUG' if param in itug_params else 'ISWAY',
                            'parameter': param,
                            'cd_measure': cd_measure,
                            'coefficient': model.params['balance_param'],
                            'p_value': model.pvalues['balance_param'],
                            'significant': model.pvalues['balance_param'] < 0.05
                        })
                    except Exception as e:
                        print(f"Error in mixed effects model for {param}, {cd_measure}: {str(e)}")
                        results.append({
                            'test_type': 'ITUG' if param in itug_params else 'ISWAY',
                            'parameter': param,
                            'cd_measure': cd_measure,
                            'coefficient': np.nan,
                            'p_value': np.nan,
                            'significant': False
                        })
                else:
                    results.append({
                        'test_type': 'ITUG' if param in itug_params else 'ISWAY',
                        'parameter': param,
                        'cd_measure': cd_measure,
                        'coefficient': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    })

    return pd.DataFrame(results)


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
    if not corr_df['correlation'].isna().all():
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
    if not mixed_df['coefficient'].isna().all():
        cd_measures = mixed_df['cd_measure'].unique()
        for cd_measure in cd_measures:
            measure_df = mixed_df[mixed_df['cd_measure'] == cd_measure]
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
                plt.title(f'Mixed Effects Model Coefficients - {cd_measure}')
                plt.xlabel('Coefficient')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'mixed_effects_forest_plot_{cd_measure}.png'))
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
    import glob

    # Function to load data from participant_summary.json files
    def load_data(base_dir):
        """
        Reads participant_summary.json files for each participant and converts to a DataFrame

        Parameters:
        -----------
        base_dir : str
            Base directory containing 'sensor_analysis' folder with participant data

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with all participant data
        """
        sensor_dir = os.path.join(base_dir, "sensor_analysis")

        df_list = []

        # Find all participant directories in sensor_analysis
        participant_dirs = glob.glob(os.path.join(sensor_dir, "*-*-*-*-*"))

        for participant_dir in participant_dirs:
            # Load the participant summary JSON file
            summary_path = os.path.join(participant_dir, "participant_summary.json")

            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        participant_data = json.load(f)

                    # Extract all data from the participant summary
                    person_df = {
                        'name': participant_data.get('name', ''),
                        'gender': participant_data.get('gender', ''),
                        'gender_code': participant_data.get('gender_code', ''),
                        'age': participant_data.get('age', 0),
                        'weight': participant_data.get('weight', 0),
                        'height': participant_data.get('height', 0),
                        'bmi': participant_data.get('bmi', 0)
                    }

                    # Extract pre-ITUG metrics
                    pre_itug = participant_data.get('pre_itug_metrics', {})
                    for key, value in pre_itug.items():
                        if key not in ['test_type', 'valid']:  # Skip non-numeric fields
                            person_df[f'pre_itug_{key}'] = value

                    # Extract post-ITUG metrics
                    post_itug = participant_data.get('post_itug_metrics', {})
                    for key, value in post_itug.items():
                        if key not in ['test_type', 'valid']:  # Skip non-numeric fields
                            person_df[f'post_itug_{key}'] = value

                    # Extract pre-ISWAY metrics
                    pre_isway = participant_data.get('pre_isway_metrics', {})
                    for key, value in pre_isway.items():
                        if key not in ['test_type', 'valid']:  # Skip non-numeric fields
                            person_df[f'pre_isway_{key}'] = value

                    # Extract post-ISWAY metrics
                    post_isway = participant_data.get('post_isway_metrics', {})
                    for key, value in post_isway.items():
                        if key not in ['test_type', 'valid']:  # Skip non-numeric fields
                            person_df[f'post_isway_{key}'] = value

                    # Extract food/cognitive dissonance metrics
                    person_df['initial_rating'] = participant_data.get('initial_rating', 0)
                    person_df['final_rating'] = participant_data.get('final_rating', 0)
                    person_df['rejected_pair_rating_diff'] = participant_data.get('rejected_pair_rating_diff', 0)
                    person_df['chosen_pair_rating_diff'] = participant_data.get('chosen_pair_rating_diff', 0)
                    person_df['computer_pair_rating_diff'] = participant_data.get('computer_pair_rating_diff', 0)
                    person_df['timeout_count'] = participant_data.get('timeout_count', 0)

                    # Calculate cognitive dissonance measures
                    person_df['average_rating_change'] = abs(person_df['final_rating'] - person_df['initial_rating'])
                    person_df['average_rejected_rating_difference'] = abs(person_df['rejected_pair_rating_diff'])

                    df_list.append(person_df)
                    print(f"Loaded data for participant: {person_df['name']}")

                except Exception as e:
                    print(f"Error loading data for {participant_dir}: {e}")

        # Combine all person data into a single DataFrame
        if df_list:
            df = pd.DataFrame(df_list)
            print(f"Successfully loaded data for {len(df)} participants")
            return df
        else:
            print("No valid participant data found!")
            return pd.DataFrame()

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
        isway_params = ['PC1_std', 'PC1_range', 'PC1_dominant_freq', 'PC1_spectral_entropy', 'sway_area',
                        'mean_distance',
                        'path_length']
        for param in isway_params:
            balance_changes[f'isway_{param}_change'] = df[f'post_isway_{param}'] - df[f'pre_isway_{param}']

        # Add cognitive dissonance measures for analysis
        balance_changes['average_rating_change'] = df['average_rating_change']
        balance_changes['average_rejected_rating_difference'] = df['average_rejected_rating_difference']
        balance_changes['chosen_pair_rating_diff'] = df['chosen_pair_rating_diff']
        balance_changes['computer_pair_rating_diff'] = df['computer_pair_rating_diff']

        # Add demographic information for potential subgroup analysis
        balance_changes['gender'] = df['gender']
        balance_changes['age'] = df['age']
        balance_changes['bmi'] = df['bmi']

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
            try:
                # Correlation with average rating change
                corr_rating, p_rating = stats.pearsonr(df[f'pre_itug_{param}'], df['average_rating_change'])

                # Correlation with rejected rating difference
                corr_rejected, p_rejected = stats.pearsonr(df[f'pre_itug_{param}'],
                                                           df['average_rejected_rating_difference'])

                # Store results
                correlation_results['test_type'].append('ITUG')
                correlation_results['parameter'].append(param)
                correlation_results['correlation_with_rating_change'].append(corr_rating)
                correlation_results['p_value_rating_change'].append(p_rating)
                correlation_results['correlation_with_rejected_diff'].append(corr_rejected)
                correlation_results['p_value_rejected_diff'].append(p_rejected)
            except Exception as e:
                print(f"Error in correlation analysis for ITUG {param}: {e}")

        # Correlations for ISWAY measures
        for param in isway_params:
            try:
                # Correlation with average rating change
                corr_rating, p_rating = stats.pearsonr(df[f'pre_isway_{param}'], df['average_rating_change'])

                # Correlation with rejected rating difference
                corr_rejected, p_rejected = stats.pearsonr(df[f'pre_isway_{param}'],
                                                           df['average_rejected_rating_difference'])

                # Store results
                correlation_results['test_type'].append('ISWAY')
                correlation_results['parameter'].append(param)
                correlation_results['correlation_with_rating_change'].append(corr_rating)
                correlation_results['p_value_rating_change'].append(p_rating)
                correlation_results['correlation_with_rejected_diff'].append(corr_rejected)
                correlation_results['p_value_rejected_diff'].append(p_rejected)
            except Exception as e:
                print(f"Error in correlation analysis for ISWAY {param}: {e}")

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
            try:
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
            except Exception as e:
                print(f"Error in regression analysis for {dep_var} using ITUG parameters: {e}")

            # ISWAY regression
            try:
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
            except Exception as e:
                print(f"Error in regression analysis for {dep_var} using ISWAY parameters: {e}")

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

        # Define parameters to analyze
        itug_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'stride_frequency']
        isway_params = ['PC1_std', 'PC1_range', 'PC1_spectral_entropy', 'sway_area', 'mean_distance', 'path_length']

        anova_results = []

        # ANOVA for ITUG parameters
        for param in itug_params:
            try:
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
            try:
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
            try:
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
            except Exception as e:
                print(f"Error in t-test for ITUG {param}: {e}")

        # T-tests for ISWAY parameters
        for param in isway_params:
            try:
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
            except Exception as e:
                print(f"Error in t-test for ISWAY {param}: {e}")

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

                try:
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
                except Exception as e:
                    print(f"Error calculating correlation between {cd_measure} and {balance_param}: {e}")

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

            # Create directory for visualizations if it doesn't exist
            os.makedirs("analysis_results", exist_ok=True)

            for _, row in significant_corrs.iterrows():
                try:
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
                    plt.savefig(f"analysis_results/correlation_{cd_measure}_{balance_param}.png")
                    plt.close()
                except Exception as e:
                    print(f"Error creating scatter plot for {cd_measure} and {balance_param}: {e}")

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
                    if not pre_row.empty:
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
                    if not pre_row.empty:
                        post_data.loc[post_data.index[idx], f'pre_{param}'] = pre_row[param].values[0]
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

                    # Additional model for rejected rating difference
                    formula3 = f"{param} ~ rejected_rating_diff + pre_{param}"
                    model3 = smf.ols(formula3, data=post_data).fit()

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
                    print(f"Error in mixed effects model for ISWAY {param}: {e}")

                return pd.DataFrame(mixed_model_results)

                def create_visualizations(results, output_dir):
                    """
                    Create and save visualizations based on analysis results
                    """
                    os.makedirs(output_dir, exist_ok=True)

                    # 1. Correlation Heatmap
                    try:
                        corr_df = results['correlations']
                        if not corr_df.empty and not corr_df['correlation'].isna().all():
                            pivot_data = corr_df.pivot_table(
                                index=['test_type', 'parameter'],
                                columns='cd_measure',
                                values='correlation'
                            )
                            plt.figure(figsize=(12, 10))
                            sns.heatmap(pivot_data, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=.5)
                            plt.title('Correlation Between Balance Parameters and Cognitive Dissonance', fontsize=16)
                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
                            plt.close()
                        else:
                            print("Skipping correlation heatmap: insufficient data.")
                    except Exception as e:
                        print(f"Error creating correlation heatmap: {e}")

                    # 2. Paired T-Test Bar Plots
                    try:
                        ttest_df = results['ttest']
                        for test_type in ['ITUG', 'ISWAY']:
                            df_subset = ttest_df[ttest_df['test_type'] == test_type]
                            if not df_subset.empty:
                                plt.figure(figsize=(12, 7))
                                bars = sns.barplot(x='parameter', y='t_statistic', data=df_subset,
                                                   palette=['green' if sig else 'gray' for sig in
                                                            df_subset['significant']])

                                # Add p-values above bars
                                for i, p in enumerate(df_subset['p_value']):
                                    plt.text(i, df_subset['t_statistic'].iloc[i] + (
                                        0.1 if df_subset['t_statistic'].iloc[i] >= 0 else -0.3),
                                             f'p={p:.3f}', ha='center')

                                plt.title(f'{test_type} Parameters: Pre vs Post Paired T-Test Results', fontsize=16)
                                plt.xticks(rotation=45, ha='right')
                                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                                plt.ylabel('T-Statistic')
                                plt.tight_layout()
                                plt.savefig(os.path.join(output_dir, f'{test_type}_ttest_barplot.png'), dpi=300)
                                plt.close()
                        else:
                            print(f"Skipping {test_type} t-test plot: insufficient data.")
                    except Exception as e:
                        print(f"Error creating t-test bar plots: {e}")

                    # 3. Mixed Effects Model Forest Plot
                    try:
                        mixed_df = results['mixed_effects']
                        if not mixed_df.empty and not mixed_df['coefficient'].isna().all():
                            # Create separate plots for each model type
                            for model_type in mixed_df['model_type'].unique():
                                model_data = mixed_df[mixed_df['model_type'] == model_type].copy()
                                if not model_data.empty:
                                    model_data['label'] = model_data.apply(
                                        lambda row: f"{row['test_type']} {row['parameter']}", axis=1)

                                    # Sort by coefficient value
                                    model_data = model_data.sort_values('coefficient')

                                    # Determine y-position for each row
                                    y_pos = range(len(model_data))

                                    plt.figure(figsize=(10, max(6, len(model_data) * 0.4)))

                                    # Plot coefficients and confidence intervals
                                    plt.scatter(model_data['coefficient'], y_pos, s=80,
                                                c=['green' if p < 0.05 else 'gray' for p in model_data['p_value']])

                                    # Add vertical line at zero
                                    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)

                                    # Add labels and title
                                    plt.yticks(y_pos, model_data['label'])

                                    model_type_labels = {
                                        'pre_balance_on_cd': 'Effect of Initial Balance on Cognitive Dissonance',
                                        'cd_on_post_balance': 'Effect of Cognitive Dissonance on Final Balance',
                                        'rejected_diff_on_post_balance': 'Effect of Rejected Rating Difference on Final Balance'
                                    }

                                    plt.title(model_type_labels.get(model_type, model_type), fontsize=16)
                                    plt.xlabel('Coefficient (Effect Size)')
                                    plt.tight_layout()
                                    plt.grid(axis='x', alpha=0.3)
                                    plt.savefig(os.path.join(output_dir, f'mixed_effects_{model_type}_plot.png'),
                                                dpi=300)
                                    plt.close()
                        else:
                            print("Skipping mixed effects forest plot: insufficient data.")
                    except Exception as e:
                        print(f"Error creating mixed effects forest plot: {e}")

                    # 4. Change Score Correlation Plot
                    try:
                        change_df = results['change_score']
                        if not change_df.empty and not change_df['correlation'].isna().all():
                            significant_changes = change_df[change_df['significant'] == True]
                            if not significant_changes.empty:
                                plt.figure(figsize=(12, 8))
                                bars = sns.barplot(x='parameter', y='correlation', hue='test_type',
                                                   data=significant_changes)

                                # Add significance markers
                                for i, p in enumerate(significant_changes['p_value']):
                                    marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                                    plt.text(i, significant_changes['correlation'].iloc[i] +
                                             (0.05 if significant_changes['correlation'].iloc[i] >= 0 else -0.1),
                                             marker, ha='center', fontsize=14)

                                plt.title('Significant Correlations Between Balance Changes and Cognitive Dissonance',
                                          fontsize=16)
                                plt.xticks(rotation=45, ha='right')
                                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                                plt.ylabel('Correlation Coefficient')
                                plt.tight_layout()
                                plt.savefig(os.path.join(output_dir, 'significant_change_correlations.png'), dpi=300)
                                plt.close()
                            else:
                                print("Skipping change score correlation plot: no significant correlations.")
                        else:
                            print("Skipping change score correlation plot: insufficient data.")
                    except Exception as e:
                        print(f"Error creating change score correlation plot: {e}")

                    # 5. Demographics visualization
                    try:
                        print("Creating demographic visualizations...")
                        demographics = pd.DataFrame({
                            'Gender': results['demographics']['gender'],
                            'Age': results['demographics']['age'],
                            'BMI': results['demographics']['bmi']
                        })

                        # Gender distribution
                        plt.figure(figsize=(8, 6))
                        gender_counts = demographics['Gender'].value_counts()
                        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,
                                colors=['skyblue', 'lightpink'])
                        plt.title('Gender Distribution', fontsize=16)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'gender_distribution.png'), dpi=300)
                        plt.close()

                        # Age distribution
                        plt.figure(figsize=(10, 6))
                        sns.histplot(demographics['Age'], kde=True, bins=10)
                        plt.title('Age Distribution', fontsize=16)
                        plt.xlabel('Age (years)')
                        plt.ylabel('Count')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=300)
                        plt.close()

                        # BMI distribution
                        plt.figure(figsize=(10, 6))
                        sns.histplot(demographics['BMI'], kde=True, bins=10)
                        plt.title('BMI Distribution', fontsize=16)
                        plt.xlabel('BMI (kg/m²)')
                        plt.ylabel('Count')

                        # Add BMI category lines
                        plt.axvline(x=18.5, color='r', linestyle='--', alpha=0.7, label='Underweight < 18.5')
                        plt.axvline(x=25, color='g', linestyle='--', alpha=0.7, label='Normal 18.5-24.9')
                        plt.axvline(x=30, color='b', linestyle='--', alpha=0.7, label='Overweight 25-29.9')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'bmi_distribution.png'), dpi=300)
                        plt.close()

                    except Exception as e:
                        print(f"Error creating demographic visualizations: {e}")

                def main():
                    """
                    Main function to run the analysis
                    """
                    # Define parameters
                    base_dir = "analysis_results"
                    output_dir = "analysis_results/output"
                    os.makedirs(output_dir, exist_ok=True)

                    # Load and preprocess data
                    print("Loading and preprocessing data...")
                    df = load_data(base_dir)

                    if df.empty:
                        print("No valid data found. Exiting.")
                        return

                    print(f"Loaded data for {len(df)} participants")

                    # Calculate balance changes
                    balance_changes = calculate_balance_changes(df)

                    # Run analyses
                    results = {
                        'demographics': {
                            'gender': df['gender'],
                            'age': df['age'],
                            'bmi': df['bmi']
                        },
                        'correlations': correlation_analysis(df),
                        'regression': linear_regression_analysis(df),
                        'anova': anova_analysis(df),
                        'ttest': paired_ttest_analysis(df),
                        'change_score': change_score_analysis(balance_changes, df),
                        'mixed_effects': mixed_effects_model(df)
                    }

                    # Create visualizations
                    create_visualizations(results, output_dir)

                    # Save results to CSV
                    print("\nSaving results to CSV files...")
                    for key, result_df in results.items():
                        if isinstance(result_df, pd.DataFrame):
                            result_df.to_csv(os.path.join(output_dir, f'{key}_results.csv'), index=False)
                            print(f"Saved {key}_results.csv")

                    # Save participant data
                    df.to_csv(os.path.join(output_dir, 'participant_data.csv'), index=False)
                    print("Saved participant_data.csv")

                    print("\nAnalysis complete. Results saved in", output_dir)

                if __name__ == "__main__":
                    main()