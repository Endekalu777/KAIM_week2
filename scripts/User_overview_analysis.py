import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from IPython.display import display

class TelecomDataAnalysis:
    def __init__(self, df):
        self.df = df
        self.user_aggregates = None

    def preprocess_data(self):
        print("Missing values before handling:")
        print(self.df.isnull().sum())

        self.handle_missing_values()
        self.detect_handle_outliers()
        self.create_user_aggregates()

        print("\nMissing values after handling:")
        print(self.df.isnull().sum())

    def handle_missing_values(self):
        # Time and duration data
        self.df['Start'].interpolate(method='linear', inplace=True)
        self.df['Start ms'].fillna(self.df['Start ms'].median(), inplace=True)
        self.df['End'].interpolate(method='linear', inplace=True)
        self.df['End ms'].fillna(self.df['End ms'].median(), inplace=True)
        self.df['Dur. (ms)'].fillna(self.df['Dur. (ms)'].median(), inplace=True)
        self.df['Activity Duration DL (ms)'].fillna(self.df['Activity Duration DL (ms)'].median(), inplace=True)
        self.df['Activity Duration UL (ms)'].fillna(self.df['Activity Duration UL (ms)'].median(), inplace=True)
        self.df['Dur. (ms).1'].fillna(self.df['Dur. (ms).1'].median(), inplace=True)

        # IMSI and MSISDN/Number
        self.df['IMSI'].fillna(self.df['IMSI'].median(), inplace=True)
        self.df['MSISDN/Number'].fillna(self.df['MSISDN/Number'].mode()[0], inplace=True)

        # IMEI and Last Location Name
        self.df['IMEI'].fillna(self.df['IMEI'].mode()[0], inplace=True)
        self.df['Last Location Name'].fillna(self.df['Last Location Name'].mode()[0], inplace=True)

        # RTT and Throughput data
        self.df['Avg RTT DL (ms)'].fillna(self.df['Avg RTT DL (ms)'].median(), inplace=True)
        self.df['Avg RTT UL (ms)'].fillna(self.df['Avg RTT UL (ms)'].median(), inplace=True)
        self.df['Avg Bearer TP DL (kbps)'].fillna(self.df['Avg Bearer TP DL (kbps)'].median(), inplace=True)
        self.df['Avg Bearer TP UL (kbps)'].fillna(self.df['Avg Bearer TP UL (kbps)'].median(), inplace=True)

        # TCP Retransmission Volumes
        self.df['TCP DL Retrans. Vol (Bytes)'].fillna(self.df['TCP DL Retrans. Vol (Bytes)'].median(), inplace=True)
        self.df['TCP UL Retrans. Vol (Bytes)'].fillna(self.df['TCP UL Retrans. Vol (Bytes)'].median(), inplace=True)

        # Percentage data
        percent_fields = [
            'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)', '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)',
            'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)'
        ]
        for field in percent_fields:
            self.df[field].fillna(self.df[field].mean(), inplace=True)

        # HTTP DL and UL data
        self.df['HTTP DL (Bytes)'].fillna(self.df['HTTP DL (Bytes)'].median(), inplace=True)
        self.df['HTTP UL (Bytes)'].fillna(self.df['HTTP UL (Bytes)'].median(), inplace=True)

        # Handset information
        self.df.dropna(subset=['Handset Manufacturer', 'Handset Type'], inplace=True)

        # NB of sec data
        nb_sec_fields = [
            'Nb of sec with 125000B < Vol DL', 'Nb of sec with 1250B < Vol UL < 6250B', 'Nb of sec with 31250B < Vol DL < 125000B',
            'Nb of sec with 37500B < Vol UL', 'Nb of sec with 6250B < Vol DL < 31250B', 'Nb of sec with 6250B < Vol UL < 37500B',
            'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B'
        ]
        for field in nb_sec_fields:
            self.df[field].fillna(self.df[field].median(), inplace=True)

        # Total Bytes data
        self.df['Total UL (Bytes)'].fillna(self.df['Total UL (Bytes)'].median(), inplace=True)
        self.df['Total DL (Bytes)'].fillna(self.df['Total DL (Bytes)'].median(), inplace=True)

    def detect_handle_outliers(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(zscore(self.df[numeric_columns]))
        outliers = (z_scores > 3)
        
        for col in numeric_columns:
            self.df.loc[outliers[col], col] = self.df[col].mean()

    def create_user_aggregates(self):
        self.df['Total Data (Bytes)'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']
        
        self.user_aggregates = self.df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',
            'Dur. (ms)': 'sum',
            'Total Data (Bytes)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum',
            'Social Media DL (Bytes)': 'sum',
            'Social Media UL (Bytes)': 'sum',
            'Google DL (Bytes)': 'sum',
            'Google UL (Bytes)': 'sum',
            'Email DL (Bytes)': 'sum',
            'Email UL (Bytes)': 'sum',
            'Youtube DL (Bytes)': 'sum',
            'Youtube UL (Bytes)': 'sum',
            'Netflix DL (Bytes)': 'sum',
            'Netflix UL (Bytes)': 'sum',
            'Gaming DL (Bytes)': 'sum',
            'Gaming UL (Bytes)': 'sum',
            'Other DL (Bytes)': 'sum',
            'Other UL (Bytes)': 'sum'
        }).reset_index()

        self.user_aggregates.rename(columns={
            'Bearer Id': 'Session Count',
            'Dur. (ms)': 'Total Duration (ms)'
        }, inplace=True)

        # Convert bytes to megabytes
        bytes_columns = [col for col in self.user_aggregates.columns if 'Bytes' in col]
        for col in bytes_columns:
            new_col_name = col.replace('Bytes', 'MB')
            self.user_aggregates[new_col_name] = self.user_aggregates[col] / (1024 * 1024)
            self.user_aggregates.drop(col, axis=1, inplace=True)

        # Convert duration to minutes
        self.user_aggregates['Total Duration (min)'] = self.user_aggregates['Total Duration (ms)'] / (1000 * 60)
        self.user_aggregates.drop('Total Duration (ms)', axis=1, inplace=True)
        # Add Total Data (DL+UL) column
        self.user_aggregates['Total Data (MB)'] = self.user_aggregates['Total DL (MB)'] + self.user_aggregates['Total UL (MB)']

    def analyze_top_users(self):
        print("Top 10 users by Total Data:")
        print(self.user_aggregates.nlargest(10, 'Total Data (MB)')[['MSISDN/Number', 'Total Data (MB)']])

        print("\nTop 10 users by Session Count:")
        print(self.user_aggregates.nlargest(10, 'Session Count')[['MSISDN/Number', 'Session Count']])

        print("\nTop 10 users by Total Duration:")
        print(self.user_aggregates.nlargest(10, 'Total Duration (min)')[['MSISDN/Number', 'Total Duration (min)']])

    def analyze_user_engagement(self):
        engagement_metrics = ['Total Data (MB)', 'Session Count', 'Total Duration (min)']
        
        print("User Engagement Statistics:")
        print(self.user_aggregates[engagement_metrics].describe())

        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(engagement_metrics, 1):
            plt.subplot(1, 3, i)
            self.user_aggregates[metric].hist(bins=50)
            plt.title(f'Distribution of {metric}')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def analyze_app_usage(self):
        app_columns = [col for col in self.user_aggregates.columns if any(app in col for app in ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other'])]
        app_usage = self.user_aggregates[app_columns].sum().sort_values(ascending=False)

        print("Application Usage Statistics:")
        print(app_usage)

        plt.figure(figsize=(12, 6))
        app_usage.plot(kind='bar')
        plt.title('Total Data Usage by Application')
        plt.xlabel('Application')
        plt.ylabel('Total Data Usage (MB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def perform_pca(self):
        features = self.user_aggregates[['Total Data (MB)', 'Session Count', 'Total Duration (min)']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)

        print("PCA Results:")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

        plt.figure(figsize=(10, 8))
        plt.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.7)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA of User Engagement Metrics')
        plt.tight_layout()
        plt.show()

    def top_handsets(self, n=10):
        return self.df['Handset Type'].value_counts().head(n)

    def top_manufacturers(self, n=3):
        return self.df['Handset Manufacturer'].value_counts().head(n)

    def top_handsets_per_manufacturer(self, n_manufacturers=3, n_handsets=5):
        top_manufacturers = self.top_manufacturers(n_manufacturers).index
        result = {}
        for manufacturer in top_manufacturers:
            handsets = self.df[self.df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(n_handsets)
            result[manufacturer] = handsets
        return result

    def describe_variables(self):
        return self.df.dtypes

    def segment_users(self):
        self.user_aggregates['Decile'] = pd.qcut(self.user_aggregates['Total Duration (min)'], q=10, labels=False)
        return self.user_aggregates.groupby('Decile')['Total Data (MB)'].sum()

    def basic_metrics(self):
        return self.df.describe()

    def dispersion_parameters(self):
        numeric_data = self.df.select_dtypes(include=[np.number])
        return numeric_data.agg(['std', 'var', 'skew', 'kurt'])

    def plot_histograms(self):
        numeric_data = self.df.select_dtypes(include=[np.number])
        fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 25))
        axes = axes.flatten()
        
        for i, column in enumerate(numeric_data.columns):
            if i < 20:  # Limit to 20 plots
                numeric_data[column].hist(ax=axes[i])
                axes[i].set_title(column)
                axes[i].set_xlabel('')
        
        plt.tight_layout()
        return fig
    
    def univariate_analysis(self, columns):
        """
        Perform univariate analysis on specified columns.
        
        Args:
        columns (list): List of column names to analyze.
        """
        # Histogram
        fig, ax = plt.subplots(figsize=(15, 5))
        self.user_aggregates[columns].hist(bins=50, ax=ax)
        plt.title('Histogram of Selected Variables')
        plt.tight_layout()
        display(fig)
        plt.close(fig)

        # Box Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=self.user_aggregates[columns], ax=ax)
        plt.title('Box Plot of Selected Variables')
        plt.tight_layout()
        display(fig)
        plt.close(fig)

    def bivariate_analysis(self):
        apps = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
        total_data = self.df['Total UL (Bytes)'] + self.df['Total DL (Bytes)']
        
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
        axes = axes.flatten()
        
        for i, app in enumerate(apps):
            x = self.df[f'{app} DL (Bytes)'] + self.df[f'{app} UL (Bytes)']
            axes[i].scatter(x, total_data)
            axes[i].set_xlabel(f'{app} Data')
            axes[i].set_ylabel('Total Data')
            axes[i].set_title(f'{app} vs Total Data')
        
        plt.tight_layout()
        return fig

    def correlation_analysis(self):
        apps = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
        corr_data = pd.DataFrame()
        
        for app in apps:
            corr_data[app] = self.df[f'{app} DL (Bytes)'] + self.df[f'{app} UL (Bytes)']
        correlation_matrix = corr_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.show()

        return corr_data.corr()

    def run_analysis(self):
        print("Preprocessing data:")
        self.preprocess_data()

        print("\nCreating user aggregates:")
        self.create_user_aggregates()
        
        print("\nAnalyzing top users:")
        self.analyze_top_users()
        
        print("\nAnalyzing user engagement:")
        self.analyze_user_engagement()
        
        print("\nAnalyzing app usage:")
        self.analyze_app_usage()
        
        print("\nTop 10 Handsets:")
        display(self.top_handsets())
        
        print("\nTop 3 Manufacturers:")
        display(self.top_manufacturers())
        
        print("\nTop Handsets per Manufacturer:")
        top_handsets_per_manufacturer = self.top_handsets_per_manufacturer()
        for manufacturer, handsets in top_handsets_per_manufacturer.items():
            print(f"\n{manufacturer}:")
            display(handsets)
        
        print("\nUser Segmentation:")
        display(self.segment_users())
        
        print("\nBasic Metrics:")
        display(self.basic_metrics())

        print("\nDescriptive Statistics:")
        display(self.describe_variables())
        
        print("\nDispersion Parameters:")
        display(self.dispersion_parameters())
        
        
        print("\nBivariate Analysis:")
        fig = self.bivariate_analysis()
        display(fig)
        plt.close(fig)
        
        print("\nCorrelation Analysis:")
        corr_matrix = self.correlation_analysis()
        display(corr_matrix)

        print("\nUnivariate Analysis:")
        self.univariate_analysis(['Total DL (MB)', 'Total UL (MB)', 'Total Data (MB)'])

        print("\nPerforming PCA:")
        self.perform_pca()

    

