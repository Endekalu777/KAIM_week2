import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
import mlflow
import mlflow.sklearn

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