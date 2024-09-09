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
from all_analysis import UserOverviewAnalysis, UserEngagement, UserExperience, UserSatisfactionAnalysis



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

    # Display results
    st.write("**Customer Metrics**")
    st.dataframe(analyzer.customer_metrics)
    st.write("**Top Users by Session Frequency**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'session_frequency'))
    st.write("**Top Users by Total Duration**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'total_duration'))
    st.write("**Top Users by Total Traffic**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'total_traffic'))

    # PCA Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(analyzer.pca_results[:, 0], analyzer.pca_results[:, 1])
    ax.set_title('PCA of Customer Metrics')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    st.pyplot(fig)

elif analysis_type == "User Experience":
    st.title("Telecom User Experience Analysis")

    # User Experience analysis
    analyzer = UserExperience(data)
    analyzer.run_analysis()

    # Display results
    st.write("**Customer Metrics**")
    st.dataframe(analyzer.customer_metrics)
    st.write("**Top Users by Average TCP Retransmission**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_tcp_retrans'))
    st.write("**Top Users by Average RTT**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_rtt'))
    st.write("**Top Users by Average Throughput**")
    st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_throughput'))

    # PCA Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(analyzer.pca_results[:, 0], analyzer.pca_results[:, 1])
    ax.set_title('PCA of Customer Metrics')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
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
    st.write("**Cluster Averages**")
    st.dataframe(analyzer.get_cluster_averages())

    # MLflow Tracking
    with st.expander("MLflow Tracking"):
        st.write("Model metrics and artifacts are tracked in MLflow.")
        st.write("MSE:", analyzer.mse)
        st.write("R-squared:", analyzer.r2)