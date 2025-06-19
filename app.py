import streamlit as st
import pandas as pd
from src.data_processor import DataProcessor
from src.visualizations import *

# Page config
st.set_page_config(
    page_title="Prime Bank Analytics Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Prime Bank Social Media Analytics Dashboard")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key (optional):", 
        type="password",
        help="Enter your OpenAI API key for advanced GPT analysis"
    )
    
    if api_key:
        st.success("✅ API Key configured")
        use_gpt = st.checkbox("Enable GPT Analysis", value=True)
    else:
        st.info("💡 Running without GPT features")
        use_gpt = False
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Upload CSV files from social media platforms and TXT files with reviews to analyze Prime Bank's online presence.")

# Initialize processor with or without API key
processor = DataProcessor(openai_api_key=api_key if use_gpt else None)

# Main content
st.markdown("### 📁 Upload Your Data Files")

# File upload section
col1, col2 = st.columns(2)

with col1:
    csv_files = st.file_uploader(
        "Upload CSV files (Facebook, Twitter, etc.)",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files containing social media data"
    )

with col2:
    txt_file = st.file_uploader(
        "Upload TXT file (Manual reviews)",
        type=['txt'],
        help="Upload a text file with reviews, one per line"
    )

# Add sample data download option
with st.expander("📝 Need sample data to test?"):
    st.markdown("""
    Download these sample files to test the dashboard:
    - [Sample CSV Data](https://example.com)
    - [Sample TXT Reviews](https://example.com)
    
    Or create test data by running:
    ```bash
    python create_test_data.py
    ```
    """)

# Process files when uploaded
if csv_files or txt_file:
    with st.spinner('Processing files...'):
        all_data = []
        
        # Process CSV files
        if csv_files:
            st.write(f"📊 Processing {len(csv_files)} CSV file(s)...")
            csv_data = processor.load_data_from_files(csv_files=csv_files)
            if not csv_data.empty:
                all_data.append(csv_data)
                st.success(f"✅ Loaded {len(csv_data)} rows from CSV files")
        
        # Process TXT file
        if txt_file:
            st.write("📝 Processing TXT file...")
            txt_data = processor.load_data_from_files(txt_files=[txt_file])
            if not txt_data.empty:
                all_data.append(txt_data)
                st.success(f"✅ Loaded {len(txt_data)} reviews from TXT file")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Process the data
            with st.spinner('Analyzing sentiment and emotions...'):
                processed_df = processor.process_all_data(combined_df)
            
            # Filter for Prime Bank mentions
            prime_df = processed_df[processed_df['prime_mentions'] > 0]
            
            st.success(f"✅ Analysis complete! Found {len(prime_df)} posts mentioning Prime Bank out of {len(processed_df)} total posts")
            
            # Display metrics
            st.header("📊 Key Metrics")
            metrics = create_summary_metrics(processed_df)
            
            col1, col2, col3, col4 = st.columns(4)
            for i, (label, value) in enumerate(metrics.items()):
                with [col1, col2, col3, col4][i]:
                    st.metric(label, value)
            
            # Display charts
            st.header("📈 Analysis")
            
            # First row of charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if len(prime_df) > 0:
                    fig = create_sentiment_pie(prime_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No Prime Bank mentions found for sentiment analysis")
            
            with col2:
                if len(prime_df) > 0:
                    fig = create_emotion_bar(prime_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No Prime Bank mentions found for emotion analysis")
            
            with col3:
                if len(prime_df) > 0:
                    fig = create_category_donut(prime_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No Prime Bank mentions found for category analysis")
            
            # Top Viral Posts
            st.header("🔥 Top Viral Posts Mentioning Prime Bank")
            
            if len(prime_df) > 0:
                top_posts = prime_df.nlargest(5, 'viral_score')[['text', 'sentiment', 'emotion', 'category', 'prime_mentions']]
                
                for idx, row in top_posts.iterrows():
                    with st.expander(f"Post #{idx+1} - {row['sentiment']} | {row['emotion']}"):
                        st.write(row['text'])
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Sentiment", row['sentiment'])
                        col2.metric("Emotion", row['emotion'])
                        col3.metric("Category", row['category'])
                        col4.metric("Mentions", row['prime_mentions'])
            else:
                st.info("No posts mentioning Prime Bank found")
            
            # Data table
            with st.expander("📋 View All Data"):
                st.dataframe(processed_df)
            
            # Download processed data
            csv = processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv,
                file_name="prime_bank_analysis.csv",
                mime="text/csv"
            )

else:
    # No files uploaded yet
    st.info("👆 Please upload CSV files and/or TXT file to begin analysis")
    
    # Show instructions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 CSV Files Should Contain:
        - A text column (text/content/message)
        - Optional: date, likes, shares
        - Can upload multiple files
        """)
    
    with col2:
        st.markdown("""
        ### 📝 TXT File Format:
        - One review per line
        - Plain text format
        - Manual reviews/comments
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Analysis Includes:
        - Sentiment (Positive/Negative)
        - Emotions (Joy/Frustration)
        - Categories (Inquiry/Complaint)
        """)