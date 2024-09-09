import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display
from sqlalchemy import create_engine
import mlflow
import mlflow.sklearn



class UserOverviewAnalysis:
    def __init__(self, df):
        self.df = df
        self.user_aggregates = None

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
        display(self.user_aggregates.nlargest(10, 'Total Data (MB)')[['MSISDN/Number', 'Total Data (MB)']])

        print("\nTop 10 users by Session Count:")
        display(self.user_aggregates.nlargest(10, 'Session Count')[['MSISDN/Number', 'Session Count']])

        print("\nTop 10 users by Total Duration:")
        display(self.user_aggregates.nlargest(10, 'Total Duration (min)')[['MSISDN/Number', 'Total Duration (min)']])

    def analyze_user_engagement(self):
        engagement_metrics = ['Total Data (MB)', 'Session Count', 'Total Duration (min)']
        
        print("User Engagement Statistics:")
        display(self.user_aggregates[engagement_metrics].describe())

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
        display(app_usage)

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
        self.user_aggregates['Decile'] = pd.qcut(self.user_aggregates['Total Duration (min)'], q=5, labels=False)
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

class UserEngagement:
    def __init__(self, df):
        self.df = df
        self.customer_metrics = None
        self.normalized_metrics = None
        self.app_usage = None
        self.pca_results = None

    def preprocess_data(self):
        self.aggregate_customer_metrics()
        self.normalize_metrics()

    def aggregate_customer_metrics(self):
        self.customer_metrics = self.df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',
            'Dur. (ms)': 'sum',
            'Total UL (Bytes)': 'sum',
            'Total DL (Bytes)': 'sum'
        }).rename(columns={
            'Bearer Id': 'session_frequency',
            'Dur. (ms)': 'total_duration',
            'Total UL (Bytes)': 'total_ul',
            'Total DL (Bytes)': 'total_dl'
        })
        self.customer_metrics['total_traffic'] = self.customer_metrics['total_ul'] + self.customer_metrics['total_dl']

    def normalize_metrics(self):
        scaler = MinMaxScaler()
        self.normalized_metrics = scaler.fit_transform(self.customer_metrics)

    def analyze_top_users(self):
        print("Top 10 customers by session frequency:")
        display(self.customer_metrics.nlargest(10, 'session_frequency'))
        print("\nTop 10 customers by total duration:")
        display(self.customer_metrics.nlargest(10, 'total_duration'))
        print("\nTop 10 customers by total traffic:")
        display(self.customer_metrics.nlargest(10, 'total_traffic'))

    def analyze_user_engagement(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.hist(self.customer_metrics['session_frequency'], bins=50)
        plt.title('Session Frequency Distribution')
        plt.subplot(132)
        plt.hist(self.customer_metrics['total_duration'], bins=50)
        plt.title('Total Duration Distribution')
        plt.subplot(133)
        plt.hist(self.customer_metrics['total_traffic'], bins=50)
        plt.title('Total Traffic Distribution')
        plt.tight_layout()
        plt.show()

    def analyze_app_usage(self):
        app_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)',
                       'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                       'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                       'Other DL (Bytes)', 'Other UL (Bytes)']
        self.app_usage = self.df.groupby('MSISDN/Number')[app_columns].sum()
        app_usage_total = self.app_usage.sum(axis=1)
        print("\nTop 10 users by total app usage:")
        display(app_usage_total.nlargest(10))

        app_totals = self.app_usage.sum()
        app_totals = app_totals.groupby(app_totals.index.str.split().str[0]).sum().sort_values(ascending=False)
        top_3_apps = app_totals.head(3)

        plt.figure(figsize=(10, 6))
        top_3_apps.plot(kind='bar')
        plt.title('Top 3 Most Used Applications')
        plt.xlabel('Application')
        plt.ylabel('Total Usage (Bytes)')
        plt.tight_layout()
        plt.show()

    def perform_pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        self.pca_results = pca.fit_transform(self.normalized_metrics)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.pca_results[:, 0], self.pca_results[:, 1])
        plt.title('PCA of Customer Metrics')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()

        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    def perform_clustering(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.customer_metrics['cluster'] = kmeans.fit_predict(self.normalized_metrics)

        cluster_stats = self.customer_metrics.groupby('cluster').agg({
            'session_frequency': ['min', 'max', 'mean', 'sum'],
            'total_duration': ['min', 'max', 'mean', 'sum'],
            'total_traffic': ['min', 'max', 'mean', 'sum']
        })
        print("\nCluster statistics:")
        display(cluster_stats)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self.customer_metrics.boxplot(column=['session_frequency', 'total_duration', 'total_traffic'], by='cluster', ax=axes)
        plt.tight_layout()
        plt.show()

        self.optimize_k(10)

    def optimize_k(self, max_k):
        distortions = []
        K = range(1, max_k+1)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(self.normalized_metrics)
            distortions.append(sum(np.min(cdist(self.normalized_metrics, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / self.normalized_metrics.shape[0])
        
        plt.figure(figsize=(10, 6))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.tight_layout()
        plt.show()


    def analyze_top_users_per_app(self):
        app_columns = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
        for app in app_columns:
            app_usage = self.df.groupby('MSISDN/Number')[[f'{app} DL (Bytes)', f'{app} UL (Bytes)']].sum()
            app_usage['total'] = app_usage[f'{app} DL (Bytes)'] + app_usage[f'{app} UL (Bytes)']
            print(f"\nTop 10 users for {app}:")
            display(app_usage.nlargest(10, 'total'))

    def run_analysis(self):
        self.preprocess_data()
        self.analyze_top_users()
        self.analyze_user_engagement()
        self.analyze_app_usage()
        self.analyze_top_users_per_app()
        self.perform_pca()
        self.perform_clustering()

    
class UserExperience:
    def __init__(self, df):
        self.df = df
        self.customer_metrics = None
        self.normalized_metrics = None
        self.app_usage = None
        self.pca_results = None

    def preprocess_data(self):
        self.aggregate_customer_metrics()
        self.normalize_metrics()

    def aggregate_customer_metrics(self):
        self.customer_metrics = self.df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Handset Type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
        }).rename(columns={
            'TCP DL Retrans. Vol (Bytes)': 'avg_tcp_retrans',
            'Avg RTT DL (ms)': 'avg_rtt',
            'Avg Bearer TP DL (kbps)': 'avg_throughput'
        })

    def normalize_metrics(self):
        scaler = MinMaxScaler()
        numerical_columns = ['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']
        self.normalized_metrics = scaler.fit_transform(self.customer_metrics[numerical_columns])

    def analyze_top_users(self):
        print("Top 10 customers by average TCP retransmission:")
        display(self.customer_metrics.nlargest(10, 'avg_tcp_retrans'))
        print("\nTop 10 customers by average RTT:")
        display(self.customer_metrics.nlargest(10, 'avg_rtt'))
        print("\nTop 10 customers by average throughput:")
        display(self.customer_metrics.nlargest(10, 'avg_throughput'))

    def analyze_metric_distributions(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.hist(self.customer_metrics['avg_tcp_retrans'], bins=50)
        plt.title('TCP Retransmission Distribution')
        plt.subplot(132)
        plt.hist(self.customer_metrics['avg_rtt'], bins=50)
        plt.title('RTT Distribution')
        plt.subplot(133)
        plt.hist(self.customer_metrics['avg_throughput'], bins=50)
        plt.title('Throughput Distribution')
        plt.tight_layout()
        plt.show()

    def analyze_handset_types(self):
        handset_counts = self.customer_metrics['Handset Type'].value_counts()
        print("\nTop 10 Handset Types:")
        display(handset_counts.head(10))

        plt.figure(figsize=(12, 6))
        handset_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Handset Types')
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def perform_pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        self.pca_results = pca.fit_transform(self.normalized_metrics)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.pca_results[:, 0], self.pca_results[:, 1])
        plt.title('PCA of Customer Metrics')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()

        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    def perform_clustering(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        numerical_columns = ['avg_tcp_retrans', 'avg_rtt', 'avg_throughput']
        self.customer_metrics['cluster'] = kmeans.fit_predict(self.normalized_metrics)

        self.cluster_stats = self.customer_metrics.groupby('cluster')[numerical_columns].agg(['min', 'max', 'mean'])
        
        return self.cluster_stats  # Return the cluster statistics

    def get_cluster_boxplot(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self.customer_metrics.boxplot(column=['avg_tcp_retrans', 'avg_rtt', 'avg_throughput'], by='cluster', ax=axes)
        plt.tight_layout()
        return fig

    def optimize_k(self, max_k):
        distortions = []
        K = range(1, max_k+1)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(self.normalized_metrics)
            distortions.append(sum(np.min(cdist(self.normalized_metrics, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / self.normalized_metrics.shape[0])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K, distortions, 'bx-')
        ax.set_xlabel('k')
        ax.set_ylabel('Distortion')
        ax.set_title('The Elbow Method showing the optimal k')
        plt.tight_layout()
        return fig

    def analyze_throughput_by_handset(self):
        throughput_by_handset = self.customer_metrics.groupby('Handset Type')['avg_throughput'].mean().sort_values(ascending=False)
        print("\nAverage Throughput by Handset Type:")
        display(throughput_by_handset.head(10))

        plt.figure(figsize=(12, 6))
        throughput_by_handset.head(10).plot(kind='bar')
        plt.title('Average Throughput by Handset Type (Top 10)')
        plt.xlabel('Handset Type')
        plt.ylabel('Average Throughput (kbps)')
        plt.tight_layout()
        plt.show()

    def run_analysis(self):
        self.preprocess_data()
        self.analyze_top_users()
        self.analyze_metric_distributions()
        self.analyze_handset_types()
        self.perform_pca()
        self.perform_clustering()
        self.analyze_throughput_by_handset()

class UserSatisfactionAnalysis:
    def __init__(self, data):
        self.data = data
        self.features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 
                         'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 
                         'TCP UL Retrans. Vol (Bytes)']
        self.scaler = StandardScaler()
        self.normalized_features = None
        self.kmeans_engagement = None
        self.kmeans_experience = None
        self.model = None
        self.mse = None
        self.r2 = None

    def preprocess_data(self):
        self.normalized_features = self.scaler.fit_transform(self.data[self.features])

    def calculate_scores(self):
        # Engagement clustering
        self.kmeans_engagement = KMeans(n_clusters=2, random_state=42)
        engagement_clusters = self.kmeans_engagement.fit_predict(self.normalized_features)

        # Experience clustering
        self.kmeans_experience = KMeans(n_clusters=2, random_state=42)
        experience_clusters = self.kmeans_experience.fit_predict(self.normalized_features)

        # Find less engaged and worst experience clusters
        less_engaged_cluster = 0 if np.mean(self.normalized_features[engagement_clusters == 0]) < np.mean(self.normalized_features[engagement_clusters == 1]) else 1
        worst_experience_cluster = 0 if np.mean(self.normalized_features[experience_clusters == 0]) > np.mean(self.normalized_features[experience_clusters == 1]) else 1

        # Calculate scores
        engagement_scores = np.linalg.norm(self.normalized_features - self.kmeans_engagement.cluster_centers_[less_engaged_cluster], axis=1)
        experience_scores = np.linalg.norm(self.normalized_features - self.kmeans_experience.cluster_centers_[worst_experience_cluster], axis=1)

        self.data['engagement_score'] = engagement_scores
        self.data['experience_score'] = experience_scores
        self.data['satisfaction_score'] = (engagement_scores + experience_scores) / 2

    def get_top_satisfied_customers(self, n=10):
        return self.data.nlargest(n, 'satisfaction_score')[['Bearer Id', 'satisfaction_score']]

    def build_regression_model(self):
        X = self.data[self.features]
        y = self.data['satisfaction_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)

    def cluster_scores(self):
        X_cluster = self.data[['engagement_score', 'experience_score']]
        kmeans = KMeans(n_clusters=2, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(X_cluster)

    def get_cluster_averages(self):
        return self.data.groupby('cluster').agg({
            'engagement_score': 'mean',
            'experience_score': 'mean',
            'satisfaction_score': 'mean'
        })

    def export_to_mysql(self, connection_string):
        engine = create_engine(connection_string)
        self.data[['Bearer Id', 'engagement_score', 'experience_score', 'satisfaction_score']].to_sql('user_satisfaction', engine, if_exists='replace', index=False)

    def track_model(self):
        with mlflow.start_run():
            mlflow.log_param("n_clusters", 2)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("mse", self.mse)
            mlflow.log_metric("r2", self.r2)
            mlflow.sklearn.log_model(self.model, "linear_regression_model")
            self.data.to_csv("user_data_with_scores.csv", index=False)
            mlflow.log_artifact("user_data_with_scores.csv")

    def analysis_run(self):
        self.preprocess_data()
        self.calculate_scores()
        print("Top 10 Satisfied Customers:")
        print(self.get_top_satisfied_customers())
        self.build_regression_model()
        print(f"\nRegression Model Results:\nMean Squared Error: {self.mse}\nR-squared Score: {self.r2}")
        self.cluster_scores()
        print("\nCluster Averages:")
        print(self.get_cluster_averages())
        self.export_to_mysql('mysql://username:password@localhost/database_name')
        self.track_model()
        print("\nAnalysis complete. Data exported to MySQL and model tracked with MLflow.")