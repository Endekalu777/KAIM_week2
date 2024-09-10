import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sqlalchemy import create_engine, text
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def preprocess_data(self):
        """Preprocess the data by normalizing the features."""
        self.normalized_features = self.scaler.fit_transform(self.data[self.features])

    def calculate_scores(self):
        """Calculate engagement, experience, and satisfaction scores for each user."""
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
        """Get the top n satisfied customers."""
        return self.data.nlargest(n, 'satisfaction_score')[['Bearer Id', 'satisfaction_score']]

    def build_regression_model(self):
        """Build a linear regression model to predict satisfaction scores."""
        X = self.data[self.features]
        y = self.data['satisfaction_score']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, y_pred)
        self.r2 = r2_score(self.y_test, y_pred)

    def cluster_scores(self):
        """Perform k-means clustering on engagement and experience scores."""
        X_cluster = self.data[['engagement_score', 'experience_score']]
        kmeans = KMeans(n_clusters=2, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(X_cluster)

    def get_cluster_averages(self):
        """Get average scores for each cluster."""
        return self.data.groupby('cluster').agg({
            'engagement_score': 'mean',
            'experience_score': 'mean',
            'satisfaction_score': 'mean'
        })

    def export_to_mysql(self, connection_string):
        """Export data to MySQL database."""
        engine = create_engine(connection_string)
        self.data[['Bearer Id', 'engagement_score', 'experience_score', 'satisfaction_score']].to_sql('user_satisfaction', engine, if_exists='replace', index=False)

    def display_mysql_query(self, connection_string, query):
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        print("MySQL Query Result:")
        print(df)

    def track_model(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("User Satisfaction Analysis")
    
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, "model")
            mlflow.log_param("model_type", type(self.model).__name__)
            mlflow.log_param("code_version", "1.0")
            mlflow.log_param("data_source", "telecom_data.csv")
            
            # Calculate predictions
            y_pred = self.model.predict(self.X_test)
            
            # Calculate and log metrics
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            
            mlflow.log_metric("mean_squared_error", mse)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mean_absolute_error", mae)
            
            print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
            print(f"MLflow experiment ID: {mlflow.active_run().info.experiment_id}")

    def plot_satisfaction_distribution(self):
        """Plot the distribution of satisfaction scores."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['satisfaction_score'], kde=True)
        plt.title('Distribution of Satisfaction Scores')
        plt.xlabel('Satisfaction Score')
        plt.ylabel('Count')
        plt.show()
        plt.savefig('satisfaction_distribution.png')
        plt.close()

    def analysis_run(self):
        """Run the entire analysis pipeline."""
        self.preprocess_data()
        self.calculate_scores()
        print("Top 10 Satisfied Customers:")
        print(self.get_top_satisfied_customers())
        self.build_regression_model()
        print(f"\nRegression Model Results:\nMean Squared Error: {self.mse}\nR-squared Score: {self.r2}")
        self.cluster_scores()
        print("\nCluster Averages:")
        print(self.get_cluster_averages())
        self.export_to_mysql('mysql+pymysql://root:7635@localhost:3306/telecom_analysis')
        print("\nMySQL Query Result:")
        self.display_mysql_query('mysql+pymysql://root:7635@localhost:3306/telecom_analysis', 
                                'SELECT * FROM user_satisfaction LIMIT 10')
        self.track_model()
        self.plot_satisfaction_distribution()
        print("\nAnalysis complete. Data exported to MySQL, model tracked with MLflow, and satisfaction distribution plotted.")

