# app.py

import streamlit as st
import pandas as pd
import os
import glob
from src.data_processor import DataProcessor
from src.insights_generator import InsightsGenerator
from src.visualizations import *

# --- Page Configuration ---
st.set_page_config(
    page_title="Prime Bank Analytics Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --- Caching Functions for Performance ---
# This decorator caches the output, so data loading and processing only happen once.
@st.cache_data
def load_and_process_data():
    """Loads all data from the 'data/uploads' directory and processes it."""
    # THIS IS THE CORRECTED PATH FOR YOUR FOLDER STRUCTURE
    DATA_DIR = 'data/uploads'
    
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    txt_files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    
    if not csv_files and not txt_files:
        return pd.DataFrame(), None # Return empty DataFrame and no insights if no files

    all_dfs = []
    
    # Read CSVs
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Could not read {os.path.basename(f)}: {e}")
            
    # Read TXTs
    for f in txt_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
            posts = content.split('\n')
            df = pd.DataFrame({
                'text': [post.strip() for post in posts if post.strip()],
                'source_file': os.path.basename(f)
            })
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Could not read {os.path.basename(f)}: {e}")

    if not all_dfs:
        return pd.DataFrame(), None
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    processor = DataProcessor()
    processed_df = processor.process_all_data(combined_df)
    
    # Generate insights
    insight_gen = InsightsGenerator()
    prime_df = processed_df[processed_df['prime_mentions'] > 0].copy()
    insights = insight_gen.generate_all_insights(processed_df, prime_df)
    
    return processed_df, insights

# --- Main Application ---
st.title("🏦 Prime Bank Social Media Analytics Dashboard")
st.markdown("This dashboard provides an analysis of social media posts and reviews mentioning Prime Bank and competitors.")

# Load data and insights using the cached function
processed_df, insights = load_and_process_data()

if processed_df.empty or insights is None:
    st.error("No valid data files were found in the 'data/uploads' directory. Please add CSV or TXT files to proceed.")
    st.stop()

# Filter for Prime Bank specific data for visualizations
prime_df = processed_df[processed_df['prime_mentions'] > 0].copy()

st.success(f"Analysis complete! Found **{len(prime_df)}** posts mentioning Prime Bank out of **{len(processed_df)}** total posts analyzed.")

# --- Display Metrics ---
st.header("📊 Key Metrics for Prime Bank")
metrics = create_summary_metrics(processed_df)
m_cols = st.columns(len(metrics))
for col, (label, value) in zip(m_cols, metrics.items()):
    col.metric(label, value)

# --- Tabbed Interface for Clarity ---
tab1, tab2, tab3, tab4 = st.tabs(["Sentiment & Emotion", "Topics & Categories", "Priority View", "Competitive Analysis"])

with tab1:
    st.subheader("Sentiment & Emotion Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_sentiment_pie(prime_df), use_container_width=True)
        with st.expander("View Sentiment Details"):
            st.markdown(f"**Positive Posts:** {insights['sentiment']['positive_context']}")
            for post in insights['sentiment']['examples']['Positive']:
                st.info(f"Example: \"{post[:100]}...\"")
            
            st.markdown(f"**Negative Posts:** {insights['sentiment']['negative_context']}")
            for post in insights['sentiment']['examples']['Negative']:
                st.warning(f"Example: \"{post[:100]}...\"")
            
            st.markdown(f"**Neutral Posts:** {insights['sentiment']['neutral_context']}")
            for post in insights['sentiment']['examples']['Neutral']:
                st.caption(f"Example: \"{post[:100]}...\"")

    with col2:
        st.plotly_chart(create_emotion_bar(prime_df), use_container_width=True)
        with st.expander("View Emotion Details"):
            st.markdown(f"**{insights['emotion']['summary']}**")
            st.markdown(f"**Recommendation:** {insights['emotion']['recommendation']}")
            for emotion, examples in insights['emotion']['examples'].items():
                st.write(f"**{emotion} Examples:**")
                for ex in examples:
                    st.info(f"\"{ex[:100]}...\"")

with tab2:
    st.subheader("Post Content Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_category_donut(prime_df), use_container_width=True)
    with col2:
        st.plotly_chart(create_word_frequency_chart(prime_df), use_container_width=True)
    
    st.plotly_chart(create_sentiment_by_category(prime_df), use_container_width=True)

with tab3:
    st.subheader("Priority & Virality Matrix")
    st.markdown("This chart helps identify posts that need urgent attention based on negative sentiment and high visibility (viral score).")
    st.plotly_chart(create_priority_matrix(prime_df), use_container_width=True)
    st.subheader("🔥 Top Viral Posts Mentioning Prime Bank")
    st.plotly_chart(create_viral_posts_chart(prime_df), use_container_width=True)

with tab4:
    st.subheader("Competitive Landscape")
    st.markdown(insights['comparison']['summary'])
    if 'comparison' in insights['comparison'] and insights['comparison']['comparison']:
        comp_data = insights['comparison']['comparison']
        comp_df = pd.DataFrame.from_dict(comp_data, orient='index').sort_values('positive_rate', ascending=False)
        comp_df.index = comp_df.index.str.replace('_', ' ').str.title()
        fig = px.bar(comp_df, x=comp_df.index, y='positive_rate', 
                     title='Positive Sentiment Rate by Bank', labels={'x': 'Bank', 'y': 'Positive Rate (%)'},
                     color='positive_rate', color_continuous_scale='RdYlGn', text_auto='.2f')
        fig.update_traces(textangle=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    
# --- Raw Data ---
with st.expander("📋 View Full Analyzed Data Table"):
    st.dataframe(processed_df)