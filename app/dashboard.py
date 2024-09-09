import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from KAIM_week2.scripts.user_satisfaction import TelecomAnalyzer 

# Set page config
st.set_page_config(page_title="Telecom Network Analytics Dashboard", layout="wide")

# Title
st.title("Telecom Network Analytics Dashboard")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Initialize the TelecomAnalyzer
    analyzer = TelecomAnalyzer(df)
    analyzer.preprocess_data()

    # Sidebar for navigation
    analysis_option = st.sidebar.selectbox(
        "Choose an analysis",
        ["Top Users", "Metric Distributions", "Handset Types", "PCA", "Clustering", "Throughput by Handset"]
    )

    if analysis_option == "Top Users":
        st.header("Top Users Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Top 10 by TCP Retransmission")
            st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_tcp_retrans'))
        
        with col2:
            st.subheader("Top 10 by RTT")
            st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_rtt'))
        
        with col3:
            st.subheader("Top 10 by Throughput")
            st.dataframe(analyzer.customer_metrics.nlargest(10, 'avg_throughput'))

    elif analysis_option == "Metric Distributions":
        st.header("Metric Distributions")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.hist(analyzer.customer_metrics['avg_tcp_retrans'], bins=50)
        ax1.set_title('TCP Retransmission Distribution')
        
        ax2.hist(analyzer.customer_metrics['avg_rtt'], bins=50)
        ax2.set_title('RTT Distribution')
        
        ax3.hist(analyzer.customer_metrics['avg_throughput'], bins=50)
        ax3.set_title('Throughput Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_option == "Handset Types":
        st.header("Handset Types Analysis")
        
        handset_counts = analyzer.customer_metrics['Handset Type'].value_counts()
        
        st.subheader("Top 10 Handset Types")
        st.dataframe(handset_counts.head(10))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        handset_counts.head(10).plot(kind='bar', ax=ax)
        ax.set_title('Top 10 Handset Types')
        ax.set_xlabel('Handset Type')
        ax.set_ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_option == "PCA":
        st.header("Principal Component Analysis")
        
        analyzer.perform_pca()
        
        # Capture the PCA plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(analyzer.pca_results[:, 0], analyzer.pca_results[:, 1])
        ax.set_title('PCA of Customer Metrics')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        st.pyplot(fig)
        
        st.write(f"Explained variance ratio: {analyzer.pca.explained_variance_ratio_}")

    elif analysis_option == "Clustering":
        st.header("Clustering Analysis")
        
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)
        cluster_stats = analyzer.perform_clustering(n_clusters=n_clusters)
        
        st.subheader("Cluster Statistics")
        st.dataframe(cluster_stats)
        
        # Capture the box plots
        fig = analyzer.get_cluster_boxplot()
        st.pyplot(fig)
        
        st.subheader("Elbow Method for Optimal k")
        elbow_fig = analyzer.optimize_k(10)
        st.pyplot(elbow_fig)

    elif analysis_option == "Throughput by Handset":
        st.header("Throughput Analysis by Handset Type")
        
        throughput_by_handset = analyzer.customer_metrics.groupby('Handset Type')['avg_throughput'].mean().sort_values(ascending=False)
        
        st.subheader("Average Throughput by Handset Type (Top 10)")
        st.dataframe(throughput_by_handset.head(10))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        throughput_by_handset.head(10).plot(kind='bar', ax=ax)
        ax.set_title('Average Throughput by Handset Type (Top 10)')
        ax.set_xlabel('Handset Type')
        ax.set_ylabel('Average Throughput (kbps)')
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.write("Please upload a CSV file to begin the analysis.")