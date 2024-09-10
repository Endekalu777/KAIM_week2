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