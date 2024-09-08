import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from IPython.display import display

class TelecomAnalyzer:
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



