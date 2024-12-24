import sys
import warnings
import traceback

# Data and plotting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, normaltest

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set global plot style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

class StreamingDataAnalysis:
    """
    A class to load and analyze streaming platform data,
    generate visualizations, and run statistical tests.
    """

    def __init__(self, csv_filepath):
        """
        Initialize the analysis by reading data from a CSV file
        and configuring plot styles.
        """
        try:
            print("Loading and initializing data...")
            self.streaming_df = pd.read_csv(csv_filepath)
            self.age_categories = ['all', '7+', '13+', '16+', '18+']

            # Configure plotting styles
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['axes.spines.top'] = False
            plt.rcParams['axes.spines.right'] = False

            # Clean and prepare data
            self.clean_and_prepare_data()

        except Exception as exc:
            print(f"Error in initialization: {str(exc)}")
            raise

    def clean_and_prepare_data(self):
        """
        Clean and prepare the data for subsequent analysis.
        Includes a check to ensure 'Type' is coded numerically for Movies (0) vs. TV Shows (1).
        """
        try:
            print("\nCleaning and preparing data...")

            # -- NEW CHECK/TRANSFORMATION FOR THE 'Type' COLUMN -- #
            # First, see what unique values the 'Type' column has:
            unique_type_values = self.streaming_df['Type'].unique()
            print("Unique 'Type' values found:", unique_type_values)

            # If 'Type' is in string form, convert "TV" or "TV Show" to 1, else 0
            if self.streaming_df['Type'].dtype == object or any(
                isinstance(val, str) for val in unique_type_values
            ):
                # Convert possible string categories to numeric 0=Movie, 1=TV
                self.streaming_df['Type'] = np.where(
                    self.streaming_df['Type'].astype(str).str.lower().str.contains('tv'),
                    1,
                    0
                )
                print("Converted 'Type' column to numeric (0=Movie, 1=TV Show).")

            # Process the Age column as an ordered categorical variable
            self.streaming_df['Age_Ordinal'] = pd.Categorical(
                self.streaming_df['Age'],
                categories=self.age_categories,
                ordered=True
            )

            # Convert Rotten Tomatoes scores into numeric form
            def convert_rt_score(score):
                try:
                    if pd.isna(score):
                        return np.nan
                    score_str = str(score).strip()
                    # If format is something like 'xx/100', parse numerator
                    if '/' in score_str:
                        numerator, _ = score_str.split('/')
                        return float(numerator)
                    # Otherwise handle percentage-like strings
                    return float(score_str.rstrip('%'))
                except:
                    return np.nan

            self.streaming_df['RT_Score'] = self.streaming_df['Rotten Tomatoes'].apply(convert_rt_score)

            # Create year categories
            self.streaming_df['Year_Category'] = pd.cut(
                self.streaming_df['Year'],
                bins=[1900, 1950, 1970, 1990, 2000, 2010, 2025],
                labels=['Pre-1950', '1950-1970', '1970-1990',
                        '1990-2000', '2000-2010', 'Post-2010']
            )

            print("Data cleaning and preparation completed successfully.")

        except Exception as exc:
            print(f"Error in data cleaning/preparation: {str(exc)}")
            raise

    def compute_descriptive_statistics(self):
        """
        Calculate a range of descriptive statistics for the dataset.
        """
        try:
            print("\nComputing descriptive statistics...")

            # Platform counts
            platform_counts = {
                'Netflix': self.streaming_df['Netflix'].sum(),
                'Disney+': self.streaming_df['Disney+'].sum()
            }

            # Age distribution by platform
            netflix_age_dist = (
                self.streaming_df[self.streaming_df['Netflix'] == 1]['Age']
                .value_counts()
                .reindex(self.age_categories)
            )
            disney_age_dist = (
                self.streaming_df[self.streaming_df['Disney+'] == 1]['Age']
                .value_counts()
                .reindex(self.age_categories)
            )

            # Rotten Tomatoes (RT) Score statistics
            netflix_rt_scores = self.streaming_df[
                (self.streaming_df['Netflix'] == 1) &
                (self.streaming_df['RT_Score'].notna())
            ]['RT_Score']

            disney_rt_scores = self.streaming_df[
                (self.streaming_df['Disney+'] == 1) &
                (self.streaming_df['RT_Score'].notna())
            ]['RT_Score']

            rt_stats = {
                'Netflix': netflix_rt_scores.describe(),
                'Disney+': disney_rt_scores.describe()
            }

            # Year distribution
            year_stats = {
                'Netflix': self.streaming_df[self.streaming_df['Netflix'] == 1]['Year'].describe(),
                'Disney+': self.streaming_df[self.streaming_df['Disney+'] == 1]['Year'].describe()
            }

            return {
                'platform_counts': platform_counts,
                'age_distribution': {
                    'Netflix': netflix_age_dist,
                    'Disney+': disney_age_dist
                },
                'rt_statistics': rt_stats,
                'year_statistics': year_stats
            }

        except Exception as exc:
            print(f"Error in descriptive statistics computation: {str(exc)}")
            raise

    def analyze_age_distribution(self):
        """
        Perform a chi-square test to analyze the age distribution across platforms.
        """
        try:
            print("\nAnalyzing age distributions...")

            # Create contingency table
            contingency_table = pd.crosstab(
                self.streaming_df['Age_Ordinal'],
                [self.streaming_df['Netflix'], self.streaming_df['Disney+']]
            )

            # Perform chi-square test
            chi2_stat, p_val, degrees_of_freedom, expected_freq = chi2_contingency(contingency_table)

            # Calculate Cramer's V
            total_n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            cramer_v_val = np.sqrt(chi2_stat / (total_n * min_dim))

            # Calculate proportions for Netflix and Disney+
            netflix_age_props = (
                self.streaming_df[self.streaming_df['Netflix'] == 1]['Age']
                .value_counts(normalize=True)
                .reindex(self.age_categories)
            )
            disney_age_props = (
                self.streaming_df[self.streaming_df['Disney+'] == 1]['Age']
                .value_counts(normalize=True)
                .reindex(self.age_categories)
            )

            return {
                'test_results': {
                    'chi2_statistic': chi2_stat,
                    'p_value': p_val,
                    'degrees_of_freedom': degrees_of_freedom,
                    'cramer_v': cramer_v_val
                },
                'proportions': {
                    'Netflix': netflix_age_props,
                    'Disney+': disney_age_props
                },
                'contingency_table': contingency_table
            }

        except Exception as exc:
            print(f"Error in age distribution analysis: {str(exc)}")
            raise

    def analyze_quality_scores(self):
        """
        Analyze Rotten Tomatoes scores using the Mann-Whitney U test.
        """
        try:
            print("\nAnalyzing RT quality scores...")

            # Get Netflix scores
            netflix_rt_scores = self.streaming_df[
                (self.streaming_df['Netflix'] == 1) &
                (self.streaming_df['RT_Score'].notna())
            ]['RT_Score']

            # Get Disney+ scores
            disney_rt_scores = self.streaming_df[
                (self.streaming_df['Disney+'] == 1) &
                (self.streaming_df['RT_Score'].notna())
            ]['RT_Score']

            # Test for normality
            _, netflix_norm_p = normaltest(netflix_rt_scores)
            _, disney_norm_p = normaltest(disney_rt_scores)

            # Perform Mann-Whitney U test
            statistic, p_val = mannwhitneyu(
                netflix_rt_scores,
                disney_rt_scores,
                alternative='two-sided'
            )

            # Calculate effect size (Z / sqrt(n))
            n_netflix = len(netflix_rt_scores)
            n_disney = len(disney_rt_scores)
            z_score = (statistic - (n_netflix * n_disney / 2)) / np.sqrt(
                n_netflix * n_disney * (n_netflix + n_disney + 1) / 12
            )
            effect_size_val = abs(z_score / np.sqrt(n_netflix + n_disney))

            return {
                'test_results': {
                    'statistic': statistic,
                    'p_value': p_val,
                    'effect_size': effect_size_val,
                    'sample_sizes': (n_netflix, n_disney)
                },
                'normality_test': {
                    'Netflix_p': netflix_norm_p,
                    'Disney_p': disney_norm_p
                },
                'descriptive_stats': {
                    'Netflix': netflix_rt_scores.describe(),
                    'Disney': disney_rt_scores.describe()
                }
            }

        except Exception as exc:
            print(f"Error in quality score analysis: {str(exc)}")
            raise

    def generate_visualizations(self):
        """
        Generate a set of plots to visualize age distributions,
        release years, quality scores, and additional segmentations,
        including an RT Score vs. Age Category analysis.
        """
        try:
            print("\nGenerating visualizations...")

            # === Figure 1: Distributions (Age, Year, Content Type) with Test Results ===
            fig_distributions, axes = plt.subplots(2, 2, figsize=(15, 10))
            ax1, ax2, ax3, ax4 = axes.flatten()

            # 1.1: Absolute numbers bar plot (Age Distribution)
            netflix_age_counts = (
                self.streaming_df[self.streaming_df['Netflix'] == 1]['Age']
                .value_counts()
                .reindex(self.age_categories)
            )
            disney_age_counts = (
                self.streaming_df[self.streaming_df['Disney+'] == 1]['Age']
                .value_counts()
                .reindex(self.age_categories)
            )

            x_positions = np.arange(len(self.age_categories))
            bar_width = 0.35

            bars_netflix = ax1.bar(
                x_positions - bar_width / 2,
                netflix_age_counts,
                bar_width,
                label='Netflix',
                color='#E50914'
            )
            bars_disney = ax1.bar(
                x_positions + bar_width / 2,
                disney_age_counts,
                bar_width,
                label='Disney+',
                color='#113CCF'
            )

            # Add value labels
            for bar_group in [bars_netflix, bars_disney]:
                for bar_item in bar_group:
                    height = bar_item.get_height()
                    ax1.text(
                        bar_item.get_x() + bar_item.get_width() / 2.0,
                        height,
                        f'{int(height)}' if not np.isnan(height) else '0',
                        ha='center', va='bottom'
                    )

            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(self.age_categories)
            ax1.set_ylabel('Number of Titles')
            ax1.set_title('1.1 Age Rating Distribution by Platform\n(Absolute Counts)', pad=20)
            ax1.legend()

            # Add chi-square test results
            age_dist_results = self.analyze_age_distribution()
            p_val = age_dist_results['test_results']['p_value']
            cramer_v_val = age_dist_results['test_results']['cramer_v']
            stats_text = (
                f'Chi-square test:\n'
                f'p-value: {p_val:.2e}\n'
                f"Cramer's V: {cramer_v_val:.3f}"
            )
            ax1.text(
                0.05, 0.95,
                stats_text,
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top'
            )

            # 1.2: Content type distribution by platform (Movies vs. TV)
            netflix_type_counts = self.streaming_df[self.streaming_df['Netflix'] == 1]['Type'].value_counts()
            disney_type_counts = self.streaming_df[self.streaming_df['Disney+'] == 1]['Type'].value_counts()

            type_data = pd.DataFrame({
                'Netflix': [
                    netflix_type_counts.get(0, 0),
                    netflix_type_counts.get(1, 0)
                ],
                'Disney+': [
                    disney_type_counts.get(0, 0),
                    disney_type_counts.get(1, 0)
                ]
            }, index=['Movies', 'TV Shows'])

            type_data.plot(kind='bar', ax=ax2, color=['#E50914', '#113CCF'])
            ax2.set_ylabel('Number of Titles')
            ax2.set_title('1.2 Content Type Distribution by Platform\n(Movies vs. TV Shows)', pad=20)
            for container in ax2.containers:
                ax2.bar_label(container)

            # 1.3: Distribution of Content by Release Year (Histogram)
            ax3.hist(
                self.streaming_df[self.streaming_df['Netflix'] == 1]['Year'],
                bins=40,  # Slightly more bins for finer detail
                alpha=0.6,
                label='Netflix',
                color='#E50914',
                density=True
            )
            ax3.hist(
                self.streaming_df[self.streaming_df['Disney+'] == 1]['Year'],
                bins=40,
                alpha=0.6,
                label='Disney+',
                color='#113CCF',
                density=True
            )
            ax3.set_xlabel('Release Year')
            ax3.set_ylabel('Density')
            ax3.set_title('1.3 Distribution of Content by Release Year', pad=20)
            ax3.legend()

            # 1.4: Titles by Year Category & Platform
            # We'll show a grouped bar chart with numeric annotations
            df_yearcat_counts = (
                self.streaming_df
                .query("Netflix == 1 or `Disney+` == 1")
                .assign(
                    Platform=lambda d: np.where(d['Netflix'] == 1, 'Netflix', 'Disney+')
                )
                .groupby(['Year_Category', 'Platform'])
                .size()
                .unstack(fill_value=0)
            )

            # Re-index to maintain the specified chronological order
            ordered_cats = ['Pre-1950', '1950-1970', '1970-1990',
                            '1990-2000', '2000-2010', 'Post-2010']
            df_yearcat_counts = df_yearcat_counts.reindex(ordered_cats)

            df_yearcat_counts.plot(kind='bar', ax=ax4, colormap='bwr')
            ax4.set_title('1.4 Titles by Year Category & Platform', pad=20)
            ax4.set_ylabel('Number of Titles')
            ax4.set_xlabel('Year_Category')
            ax4.legend(title='Platform')

            for container in ax4.containers:
                ax4.bar_label(container, label_type='center')

            plt.tight_layout()

            # === Figure 2: Quality Score Analysis ===
            fig_quality, axs_quality = plt.subplots(2, 3, figsize=(20, 10))
            ax_q1, ax_q2, ax_q3, ax_q4, ax_q5, ax_q6 = axs_quality.flatten()

            # 2.1: Box + Violin plots
            netflix_rt_scores = self.streaming_df[
                (self.streaming_df['Netflix'] == 1) &
                (self.streaming_df['RT_Score'].notna())
            ]['RT_Score']
            disney_rt_scores = self.streaming_df[
                (self.streaming_df['Disney+'] == 1) &
                (self.streaming_df['RT_Score'].notna())
            ]['RT_Score']

            violin_parts = ax_q1.violinplot(
                [netflix_rt_scores, disney_rt_scores],
                showmeans=True, showextrema=True
            )

            # Customize violin colors
            violin_parts['bodies'][0].set_facecolor('#E50914')
            violin_parts['bodies'][1].set_facecolor('#113CCF')
            violin_parts['bodies'][0].set_alpha(0.3)
            violin_parts['bodies'][1].set_alpha(0.3)

            # Add box plots
            box_plots = ax_q1.boxplot(
                [netflix_rt_scores, disney_rt_scores],
                positions=[1, 2],
                widths=0.15,
                patch_artist=True,
                showfliers=False
            )
            box_plots['boxes'][0].set_facecolor('#E50914')
            box_plots['boxes'][1].set_facecolor('#113CCF')

            # Statistical annotation
            quality_score_results = self.analyze_quality_scores()
            mann_whitney_p_value = quality_score_results['test_results']['p_value']
            effect_size_val = quality_score_results['test_results']['effect_size']

            annotation_text = (
                f"Mann-Whitney U Test:\n"
                f"p-value: {mann_whitney_p_value:.2e}\n"
                f"Effect size: {effect_size_val:.3f}"
            )

            ax_q1.text(
                0.005, 0.95,
                annotation_text,
                transform=ax_q1.transAxes,
                fontsize=7,
                bbox=dict(facecolor='white', edgecolor='black',
                          boxstyle='round', alpha=0.8, pad=0.5),
                verticalalignment='top',
                horizontalalignment='left'
            )

            ax_q1.set_xticks([1, 2])
            ax_q1.set_xticklabels(['Netflix', 'Disney+'])
            ax_q1.set_ylabel('Rotten Tomatoes Score')
            ax_q1.set_title('2.1 Distribution of RT Scores\n(Mann-Whitney Summary)', pad=20)

            # 2.2: Density plot
            sns.kdeplot(data=netflix_rt_scores, ax=ax_q2,
                        label='Netflix', color='#E50914', alpha=0.6)
            sns.kdeplot(data=disney_rt_scores, ax=ax_q2,
                        label='Disney+', color='#113CCF', alpha=0.6)

            ax_q2.set_xlabel('Rotten Tomatoes Score')
            ax_q2.set_ylabel('Density')
            ax_q2.set_title('2.2 Score Distribution Density', pad=20)
            ax_q2.legend()

            # 2.3: Score distributions by year (Scatter)
            netflix_filtered_df = self.streaming_df[self.streaming_df['Netflix'] == 1]
            disney_filtered_df = self.streaming_df[self.streaming_df['Disney+'] == 1]

            ax_q3.scatter(netflix_filtered_df['Year'], netflix_filtered_df['RT_Score'],
                          alpha=0.3, color='#E50914', label='Netflix')
            ax_q3.scatter(disney_filtered_df['Year'], disney_filtered_df['RT_Score'],
                          alpha=0.3, color='#113CCF', label='Disney+')

            ax_q3.set_xlabel('Release Year')
            ax_q3.set_ylabel('RT Score')
            ax_q3.set_title('2.3 Scores by Release Year', pad=20)
            ax_q3.legend()

            # 2.4: Score distribution by content type (Movies vs. TV)
            netflix_movies = netflix_filtered_df[netflix_filtered_df['Type'] == 0]['RT_Score']
            netflix_tv = netflix_filtered_df[netflix_filtered_df['Type'] == 1]['RT_Score']
            disney_movies = disney_filtered_df[disney_filtered_df['Type'] == 0]['RT_Score']
            disney_tv = disney_filtered_df[disney_filtered_df['Type'] == 1]['RT_Score']

            box_plots2 = ax_q4.boxplot(
                [netflix_movies, netflix_tv, disney_movies, disney_tv],
                labels=['Netflix\nMovies', 'Netflix\nTV', 'Disney+\nMovies', 'Disney+\nTV'],
                patch_artist=True
            )
            # Customize box colors
            color_list = ['#E50914', '#E50914', '#113CCF', '#113CCF']
            for patch, color in zip(box_plots2['boxes'], color_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax_q4.set_ylabel('Rotten Tomatoes Score')
            ax_q4.set_title('2.4 Score Distribution by Content Type', pad=20)

            # 2.5: Mean RT Scores by Year Category & Platform
            df_platform_scores = (
                self.streaming_df
                .query("RT_Score == RT_Score and (Netflix == 1 or `Disney+` == 1)")
                .assign(
                    Platform=lambda d: np.where(d['Netflix'] == 1, 'Netflix', 'Disney+')
                )
                .groupby(['Year_Category', 'Platform'])['RT_Score']
                .mean()
                .unstack(fill_value=0)
            )

            df_platform_scores.plot(kind='bar', ax=ax_q5, colormap='Set2')
            ax_q5.set_title('2.5 Mean RT Scores by Year Category & Platform', pad=20)
            ax_q5.set_ylabel('Mean RT Score')
            ax_q5.set_xlabel('Year Category')
            ax_q5.legend()

            # 2.6: RT Scores by Age Category
            age_scores_data = []
            labels_for_age = []
            for cat in self.age_categories:
                cat_scores = self.streaming_df[self.streaming_df['Age_Ordinal'] == cat]['RT_Score'].dropna()
                # Only add categories that have actual data
                if len(cat_scores) > 0:
                    age_scores_data.append(cat_scores)
                    labels_for_age.append(cat)

            if age_scores_data:
                ax_q6.boxplot(age_scores_data, labels=labels_for_age)
                ax_q6.set_title('2.6 RT Scores by Age Category', pad=20)
                ax_q6.set_xlabel('Age Category')
                ax_q6.set_ylabel('Rotten Tomatoes Score')
            else:
                ax_q6.set_title('2.6 No RT Score data for Age Categories')
                ax_q6.axis('off')

            plt.tight_layout()

            return fig_distributions, fig_quality

        except Exception as exc:
            print(f"Error in visualization creation: {str(exc)}")
            traceback.print_exc()
            raise

    def run_full_analysis_and_save(self, output_dir='.' ):
        """
        Run all analyses, generate plots, and save the outputs to files.
        """
        try:
            print("\nRunning complete analysis...")

            # Run statistical analyses
            descriptive_stats = self.compute_descriptive_statistics()
            age_distribution_results = self.analyze_age_distribution()
            quality_score_results = self.analyze_quality_scores()

            # Generate and save plots
            fig_distributions, fig_quality_scores = self.generate_visualizations()
            fig_distributions.savefig(
                f'{output_dir}/distributions_analysis.png',
                dpi=300, bbox_inches='tight'
            )
            fig_quality_scores.savefig(
                f'{output_dir}/quality_scores_analysis.png',
                dpi=300, bbox_inches='tight'
            )

            # Print key results
            print("\nKey Results Summary:")

            print("\n1. Age Distribution Analysis:")
            print(f"Chi-square test p-value: {age_distribution_results['test_results']['p_value']:.4f}")
            print(f"Cramer's V: {age_distribution_results['test_results']['cramer_v']:.4f}")

            print("\n2. Quality Score Analysis:")
            print(f"Mann-Whitney U test p-value: {quality_score_results['test_results']['p_value']:.4f}")
            print(f"Effect size: {quality_score_results['test_results']['effect_size']:.4f}")

            # Close all figures to free memory
            plt.close('all')

            return {
                'descriptive_statistics': descriptive_stats,
                'age_analysis': age_distribution_results,
                'quality_analysis': quality_score_results
            }

        except Exception as exc:
            print(f"Error in saving results: {str(exc)}")
            raise


def main(csv_filepath):
    """
    Main function to run the analysis from a CSV file path.
    """
    try:
        # Initialize and run analysis
        analyzer = StreamingDataAnalysis(csv_filepath)
        analysis_results = analyzer.run_full_analysis_and_save()
        return analysis_results

    except Exception as exc:
        print(f"Error in main execution: {str(exc)}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)

    csv_input_path = sys.argv[1]
    main(csv_input_path)
