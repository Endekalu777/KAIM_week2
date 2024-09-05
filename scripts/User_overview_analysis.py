import numpy as np
import pandas as pd
from scipy.stats import zscore
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        """Check for missing values in the DataFrame."""
        self.df.head(5)

        print("Showing missing values")

        print(self.df.isnull().sum())

    def handle_missing_values(self):
        """Handle missing values in the DataFrame by filling or dropping them appropriately."""
        # Fill missing values for time and duration data
        self.df['Start'].interpolate(method='linear', inplace=True)
        self.df['Start ms'].fillna(self.df['Start ms'].median(), inplace=True)
        self.df['End'].interpolate(method='linear', inplace=True)
        self.df['End ms'].fillna(self.df['End ms'].median(), inplace=True)
        self.df['Dur. (ms)'].fillna(self.df['Dur. (ms)'].median(), inplace=True)
        self.df['Activity Duration DL (ms)'].fillna(self.df['Activity Duration DL (ms)'].median(), inplace=True)
        self.df['Activity Duration UL (ms)'].fillna(self.df['Activity Duration UL (ms)'].median(), inplace=True)
        self.df['Dur. (ms).1'].fillna(self.df['Dur. (ms).1'].median(), inplace=True)

        # Fill missing values for IMSI and MSISDN/Number
        self.df['IMSI'].fillna(self.df['IMSI'].median(), inplace=True)
        self.df['MSISDN/Number'].fillna(self.df['MSISDN/Number'].mode()[0], inplace=True)  # Mode for categorical-like ID

        # Fill missing values for IMEI and Last Location Name
        self.df['IMEI'].fillna(self.df['IMEI'].mode()[0], inplace=True)  # Mode for categorical-like ID
        self.df['Last Location Name'].fillna(self.df['Last Location Name'].mode()[0], inplace=True)  # Mode for categorical-like location

        # Fill missing values for RTT and Throughput data
        self.df['Avg RTT DL (ms)'].fillna(self.df['Avg RTT DL (ms)'].median(), inplace=True)
        self.df['Avg RTT UL (ms)'].fillna(self.df['Avg RTT UL (ms)'].median(), inplace=True)
        self.df['Avg Bearer TP DL (kbps)'].fillna(self.df['Avg Bearer TP DL (kbps)'].median(), inplace=True)
        self.df['Avg Bearer TP UL (kbps)'].fillna(self.df['Avg Bearer TP UL (kbps)'].median(), inplace=True)

        # Fill missing values for TCP Retransmission Volumes
        self.df['TCP DL Retrans. Vol (Bytes)'].fillna(self.df['TCP DL Retrans. Vol (Bytes)'].median(), inplace=True)
        self.df['TCP UL Retrans. Vol (Bytes)'].fillna(self.df['TCP UL Retrans. Vol (Bytes)'].median(), inplace=True)

        # Fill missing values for percentage data
        percent_fields = [
            'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)', '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)',
            'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)'
        ]
        for field in percent_fields:
            self.df[field].fillna(self.df[field].mean(), inplace=True)

        # Handle HTTP DL and UL data
        self.df['HTTP DL (Bytes)'].fillna(self.df['HTTP DL (Bytes)'].median(), inplace=True)
        self.df['HTTP UL (Bytes)'].fillna(self.df['HTTP UL (Bytes)'].median(), inplace=True)

        # Fill missing values for handset information
        self.df.dropna(subset=['Handset Manufacturer'], inplace=True)
        self.df.dropna(subset=['Handset Type'], inplace=True)

        # Fill missing values for NB of sec data
        nb_sec_fields = [
            'Nb of sec with 125000B < Vol DL', 'Nb of sec with 1250B < Vol UL < 6250B', 'Nb of sec with 31250B < Vol DL < 125000B',
            'Nb of sec with 37500B < Vol UL', 'Nb of sec with 6250B < Vol DL < 31250B', 'Nb of sec with 6250B < Vol UL < 37500B',
            'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B'
        ]
        for field in nb_sec_fields:
            self.df[field].fillna(self.df[field].median(), inplace=True)

        # Fill missing values for Total Bytes data
        self.df['Total UL (Bytes)'].fillna(self.df['Total UL (Bytes)'].median(), inplace=True)
        self.df['Total DL (Bytes)'].fillna(self.df['Total DL (Bytes)'].median(), inplace=True)

        print("Missing values handled successfully.")
        print(self.df.isnull().sum())
        return self.df


class OverviewAnalysis:
    def __init__(self, df):
        self.df = df

    def top_handset_manufacturer(self):
        """Identify the top handset manufacturers and their top 5 handsets."""
        handset_usage = self.df['Handset Type'].value_counts()
        top10_handsets = handset_usage.nlargest(10)

        handset_manufacturer = self.df['Handset Manufacturer'].value_counts()
        top3_manufacturers = handset_manufacturer.nlargest(3)

        # Initialize dictionary to hold top 5 handsets per manufacturer
        top_5_per_manufacturer = {}

        # Iterate over top 3 manufacturers
        for manufacturer in top3_manufacturers.index:
            manufacturer_data = self.df[self.df['Handset Manufacturer'] == manufacturer]
            handsets = manufacturer_data['Handset Type'].value_counts()
            top_5_handsets = handsets.head(5)
            top_5_per_manufacturer[manufacturer] = top_5_handsets

        # Display top 5 handsets per manufacturer
        for manufacturer, handsets in top_5_per_manufacturer.items():
            print(f"Top 5 handsets for {manufacturer}:")
            print(handsets)


class Analysis():
    def __init__(self, df):
        self.df = df
    
    def detect_handle_outliers(self):
        # Apply Z-score
        z_scores = np.abs(zscore(self.df.select_dtypes(include=[np.number])))  # Only for numeric columns
        outliers = (z_scores > 3)

        # Replace outliers with the mean of that column
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.df.loc[outliers[
                col], col] = self.df[col].mean()
            
        return self.df
    
    def top5_deciles(self):
        self.df['Total Data (DL+UL) (Bytes)'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']
        self.df['Total Duration'] = self.df['Activity Duration DL (ms)'] + self.df['Activity Duration UL (ms)']

        # Segment users into deciles based on total duration
        self.df['Decile Class'] = pd.qcut(self.df['Total Duration'], 10, labels=False) + 1
        # Extract top 5 deciles (decile values 5-9)
        top5_deciles = self.df[self.df['Decile Class'] >= 5]
        return top5_deciles
    
    def metrics_dispersion_analysis(self):
        metrics = self.df[['Total DL (Bytes)', 'Total UL (Bytes)', 'Total Data (DL+UL) (Bytes)']].agg(['mean', 'median', 'std', 'min', 'max'])
        dispersion = self.df[['Total DL (Bytes)', 'Total UL (Bytes)', 'Total Data (DL+UL) (Bytes)']].agg(['mean', 'std', 'var', 'min', 'max'])
        return metrics, dispersion

class UserBehaviour:
    def __init__(self, df):
        self.df = df

    def user_aggregates(self):
        self.df['Total Data (DL+UL) (Bytes)'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']
        # Group by 'IMSI' to aggregate per user
        user_aggregates = self.df.groupby('IMSI').agg({
            'Bearer Id': 'count', 
            'Dur. (ms)': 'sum',    
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
            'Other UL (Bytes)': 'sum',
            'Total Data (DL+UL) (Bytes)': 'sum'
        }).reset_index()

        # Rename columns for clarity
        user_aggregates.rename(columns={
            'Bearer Id': 'Number of xDR Sessions',
            'Dur. (ms)': 'Total Session Duration (ms)'
        }, inplace=True)

        # Calculate total data volume for each application
        user_aggregates['Total Data Volume (Bytes)'] = user_aggregates[
            ['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)']
        ].sum(axis=1)
    
        # Convert Durations from ms to seconds
        user_aggregates['Total Session Duration (s)'] = user_aggregates['Total Session Duration (ms)'] / 1000
        user_aggregates.drop('Total Session Duration (ms)', axis=1, inplace=True)

        # Convert Data Sizes from Bytes to Megabytes
        bytes_to_mb = 1 / 1_048_576
        size_columns = [
            'Total DL (Bytes)', 'Total UL (Bytes)',
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)', 'Total Data Volume (Bytes)', 'Total Data (DL+UL) (Bytes)'
        ]

        # Convert bytes columns to megabytes
        for col in size_columns:
            user_aggregates[col.replace('Bytes', 'MB')] = user_aggregates[col] * bytes_to_mb
            user_aggregates.drop(col, axis=1, inplace=True)

        return user_aggregates

    
class plots():
    def __init__(self, user_aggregates):
        self.user_aggregates = user_aggregates

    def univariate_plot(self):
        # Histogram
        self.user_aggregates[['Total DL (MB)', 'Total UL (MB)', 'Total Data (DL+UL) (MB)']].hist(bins=50, figsize=(15, 5))
        plt.show()

        # Box Plot
        sns.boxplot(data=self.user_aggregates[['Total DL (MB)', 'Total UL (MB)', 'Total Data (DL+UL) (MB)']])
        plt.show()

    def scatter_plot(self):
        # Scatter plots
        sns.pairplot(self.user_aggregates[['Social Media DL (MB)', 'Google DL (MB)', 'Email DL (MB)', 
                        'Youtube DL (MB)', 'Netflix DL (MB)', 'Gaming DL (MB)', 
                        'Other DL (MB)', 'Total Data (DL+UL) (MB)']])
        plt.show()

    def correlation_matrix(self):
        correlation_matrix = self.user_aggregates[['Social Media DL (MB)', 'Google DL (MB)', 'Email DL (MB)', 
                         'Youtube DL (MB)', 'Netflix DL (MB)', 'Gaming DL (MB)', 
                         'Other DL (MB)', 'Total Data (DL+UL) (MB)']].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.show()

class PCA_analysis():
    def __init__(self, user_aggregates):
        self.user_aggregates = user_aggregates
        self.traffic_columns = None
        self.df_traffic = None
        self.top_10_per_app = None
        self.cluster_results = None

    def compute_PCA(self):
        # Standardize the data
        features = self.user_aggregates[['Total DL (MB)', 'Total UL (MB)', 'Total Data (DL+UL) (MB)']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Apply PCA
        pca = PCA(n_components=2)  # Reduce to 2 components for visualization
        pca_features = pca.fit_transform(scaled_features)

        # Explained variance
        print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
        print(f'PCA components: {pca.components_}')

        # Plot PCA results
        plt.scatter(pca_features[:, 0], pca_features[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of User Data')
        plt.show()


