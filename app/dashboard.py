import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from IPython.display import display
import mlflow
import mlflow.sklearn

# Adjust the path to include the parent directory of `scripts`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Now you should be able to import from `scripts.all_analysis`
from All_analysis import UserOverviewAnalysis, UserEngagement, UserExperience, UserSatisfactionAnalysis



st.set_page_config(
    page_title="Telecom Data Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# Load your data
data = pd.read_csv("data/cleaned_data.csv")

# Create sidebar options
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["User Overview", "User Engagement", "User Experience", "User Satisfaction"])

# Create a placeholder for the main content
main_content = st.empty()

if analysis_type == "User Overview":
    st.title("Telecom User Overview")

    # User Overview analysis
    analyzer = UserOverviewAnalysis(data)
    analyzer.run_analysis()

    # Display results
    st.write("**Top Users**")
    st.dataframe(analyzer.user_aggregates.nlargest(10, 'Total Data (MB)')[['MSISDN/Number', 'Total Data (MB)']])
    st.write("**Top Handsets**")
    st.dataframe(analyzer.top_handsets())
    st.write("**Top Manufacturers**")
    st.dataframe(analyzer.top_manufacturers())

elif analysis_type == "User Engagement":
    st.title("Telecom User Engagement Analysis")

    # User Engagement analysis
    analyzer = UserEngagement(data)
    analyzer.run_analysis()

    st.write("**Top Users by Session Frequency**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'session_frequency'))
    st.write("**Top Users by Total Duration**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'total_duration'))
    st.write("**Top Users by Total Traffic**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'total_traffic'))


elif analysis_type == "User Experience":
    st.title("Telecom User Experience Analysis")

    # User Experience analysis
    analyzer = UserExperience(data)
    analyzer.run_analysis()

    # Display results
    st.write("**Top Users by Average TCP Retransmission**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_tcp_retrans'))
    st.write("**Top Users by Average RTT**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_rtt'))
    st.write("**Top Users by Average Throughput**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_throughput'))

     # Metric Distributions
    st.subheader("Metric Distributions")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    ax1.hist(analyzer.customer_metrics['avg_tcp_retrans'], bins=50)
    ax1.set_title('TCP Retransmission Distribution')
    
    ax2.hist(analyzer.customer_metrics['avg_rtt'], bins=50)
    ax2.set_title('RTT Distribution')
    
    ax3.hist(analyzer.customer_metrics['avg_throughput'], bins=50)
    ax3.set_title('Throughput Distribution')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Handset Types Analysis
    st.subheader("Top 10 Handset Types")
    handset_counts = analyzer.customer_metrics['Handset Type'].value_counts()
    st.dataframe(handset_counts.head(10))

    fig, ax = plt.subplots(figsize=(12, 6))
    handset_counts.head(10).plot(kind='bar', ax=ax)
    ax.set_title('Top 10 Handset Types')
    ax.set_xlabel('Handset Type')
    ax.set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig)

    # Throughput by Handset Type
    st.subheader("Average Throughput by Handset Type (Top 10)")
    throughput_by_handset = analyzer.customer_metrics.groupby('Handset Type')['avg_throughput'].mean().sort_values(ascending=False)
    st.dataframe(throughput_by_handset.head(10))

    fig, ax = plt.subplots(figsize=(12, 6))
    throughput_by_handset.head(10).plot(kind='bar', ax=ax)
    ax.set_title('Average Throughput by Handset Type (Top 10)')
    ax.set_xlabel('Handset Type')
    ax.set_ylabel('Average Throughput (kbps)')
    plt.tight_layout()
    st.pyplot(fig)

    # Cluster Boxplot
    fig = analyzer.get_cluster_boxplot()
    st.pyplot(fig)

elif analysis_type == "User Satisfaction":
    st.title("Telecom User Satisfaction Analysis")

    # User Satisfaction analysis
    analyzer = UserSatisfactionAnalysis(data)
    analyzer.analysis_run()

    # Display results
    st.write("**Top 10 Satisfied Customers**")
    st.dataframe(analyzer.get_top_satisfied_customers())

    # Scatter plot of Engagement vs Experience scores
    st.subheader("Engagement vs Experience Scores")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(analyzer.data['engagement_score'], 
                         analyzer.data['experience_score'], 
                         c=analyzer.data['cluster'], 
                         cmap='viridis', 
                         alpha=0.6)
    ax.set_xlabel('Engagement Score')
    ax.set_ylabel('Experience Score')
    ax.set_title('Engagement vs Experience Scores')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    # Distribution of Satisfaction Scores
    st.subheader("Distribution of Satisfaction Scores")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(analyzer.data['satisfaction_score'], kde=True, ax=ax)
    ax.set_xlabel('Satisfaction Score')
    ax.set_title('Distribution of Satisfaction Scores')
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': analyzer.features,
        'importance': np.abs(analyzer.model.coef_)
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
    ax.set_title('Feature Importance in Predicting Satisfaction Score')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = analyzer.data[analyzer.features + ['satisfaction_score']].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    # Actual vs Predicted Satisfaction Scores
    st.subheader("Actual vs Predicted Satisfaction Scores")
    X = analyzer.data[analyzer.features]
    y = analyzer.data['satisfaction_score']
    y_pred = analyzer.model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Satisfaction Score')
    ax.set_ylabel('Predicted Satisfaction Score')
    ax.set_title('Actual vs Predicted Satisfaction Scores')
    st.pyplot(fig)