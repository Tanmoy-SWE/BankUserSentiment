# src/visualizations.py

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import Counter
import re

def create_sentiment_pie(df):
    """Create sentiment distribution pie chart"""
    sentiment_counts = df['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>')
    return fig

def create_emotion_bar(df):
    """Create emotion distribution bar chart"""
    emotion_counts = df['emotion'].value_counts()
    color_map = {'Joy': '#f39c12', 'Frustration': '#e74c3c', 'Confusion': '#3498db', 'Anxiety': '#9b59b6', 'Neutral': '#95a5a6'}
    fig = px.bar(
        x=emotion_counts.index, y=emotion_counts.values, title="Emotion Detection",
        labels={'x': 'Emotion', 'y': 'Count'}, color=emotion_counts.index, color_discrete_map=color_map
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45, yaxis=dict(gridcolor='rgba(0,0,0,0.1)'))
    return fig

def create_category_donut(df):
    """Create post category donut chart"""
    category_counts = df['category'].value_counts()
    color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig = px.pie(
        values=category_counts.values, names=category_counts.index, title="Post Categories",
        hole=0.4, color_discrete_sequence=color_sequence
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>')
    fig.add_annotation(text=f"Total<br>{len(df)}", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=20))
    return fig

def create_mentions_timeline(df):
    """Create timeline of Prime Bank mentions if date column exists"""
    date_columns = ['date', 'created_at', 'timestamp', 'Date', 'post_date']
    date_col = None
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    if not date_col: return None
    try:
        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        df_valid = df[df['date_parsed'].notna()]
        if len(df_valid) == 0: return None
        timeline_df = df_valid.groupby(df_valid['date_parsed'].dt.date).agg({'prime_mentions': 'sum', 'sentiment': lambda x: (x == 'Positive').sum()}).reset_index()
        timeline_df.columns = ['date', 'mentions', 'positive_posts']
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=timeline_df['date'], y=timeline_df['mentions'], name='Total Mentions', line=dict(color='#3498db', width=3), mode='lines+markers'), secondary_y=False)
        fig.add_trace(go.Scatter(x=timeline_df['date'], y=timeline_df['positive_posts'], name='Positive Posts', line=dict(color='#2ecc71', width=2, dash='dot'), mode='lines+markers'), secondary_y=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Number of Mentions", secondary_y=False)
        fig.update_yaxes(title_text="Positive Posts", secondary_y=True)
        fig.update_layout(title="Prime Bank Mentions Over Time", hovermode='x unified', showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig
    except Exception as e:
        print(f"Error creating timeline: {e}")
        return None

def create_viral_posts_chart(df, top_n=10):
    """Create horizontal bar chart of most viral posts"""
    if 'viral_score' not in df.columns or df.empty: return None
    top_viral = df.nlargest(top_n, 'viral_score')
    top_viral['text_truncated'] = top_viral['text'].apply(lambda x: x[:50] + '...' if len(str(x)) > 50 else x)
    fig = px.bar(
        top_viral, x='viral_score', y='text_truncated', orientation='h', title=f'Top {top_n} Viral Posts',
        color='sentiment', color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'},
        hover_data=['text', 'emotion', 'category']
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="Viral Score", yaxis_title="Post Preview", showlegend=True)
    return fig

# --- MODIFIED FUNCTION ---
def create_summary_metrics(df):
    """Calculate summary metrics for display, including new scores."""
    if 'prime_mentions' not in df.columns or df.empty:
        return {
            'Bank Sentiment Score': 0,
            'Engagement-Weighted Sentiment': '0.00'
        }

    prime_df = df[df['prime_mentions'] > 0]
    
    if not prime_df.empty:
        positive_mentions = (prime_df['sentiment'] == 'Positive').sum()
        negative_mentions = (prime_df['sentiment'] == 'Negative').sum()
        
        # New Metric 1: Bank Sentiment Score
        bank_sentiment_score = positive_mentions - negative_mentions
        
        # New Metric 2: Engagement-Weighted Sentiment
        ew_sentiment = (prime_df['polarity'] * prime_df['viral_score']).sum()
    else:
        bank_sentiment_score = 0
        ew_sentiment = 0

    metrics = {
        'Bank Sentiment Score': f"{bank_sentiment_score:+,}", # Add sign
        'Engagement-Weighted Sentiment': f"{ew_sentiment:,.2f}"
    }
    return metrics

# --- NEW FUNCTION 1 ---
def create_bank_comparison_chart(df):
    """Create bar chart comparing mentions of Prime Bank vs competitors."""
    if 'all_banks_mentioned' not in df.columns or df.empty:
        return None
        
    mentions = df['all_banks_mentioned'].explode().dropna()
    
    if mentions.empty:
        return None
        
    bank_counts = mentions.value_counts().reset_index()
    bank_counts.columns = ['Bank', 'Mentions']
    
    bank_counts['Bank'] = bank_counts['Bank'].str.replace('_', ' ').str.title()
    
    fig = px.bar(
        bank_counts, x='Bank', y='Mentions', title='Bank Mention Comparison',
        color='Bank', text='Mentions'
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Total Mentions", showlegend=False)
    fig.update_traces(textposition='outside')
    return fig

# --- NEW FUNCTION 2 ---
def create_geolocation_map(df, mapbox_token=None):
    """Create a map showing where Prime Bank mentions are coming from."""
    if 'location' not in df.columns or df.empty:
        st.info("No 'location' column found in data to generate map.")
        return None
    
    geo_mapping = {
        'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
        'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
        'Sylhet': {'lat': 24.8949, 'lon': 91.8687},
        'Rajshahi': {'lat': 24.3745, 'lon': 88.6042},
        'Khulna': {'lat': 22.8456, 'lon': 89.5403},
        'Barisal': {'lat': 22.7010, 'lon': 90.3535},
        'Rangpur': {'lat': 25.7439, 'lon': 89.2752},
    }
    
    df_loc = df.copy()
    df_loc['lat'] = df_loc['location'].map(lambda x: geo_mapping.get(x, {}).get('lat'))
    df_loc['lon'] = df_loc['location'].map(lambda x: geo_mapping.get(x, {}).get('lon'))
    
    df_loc.dropna(subset=['lat', 'lon'], inplace=True)
    
    if df_loc.empty:
        st.info("No valid locations found in data to plot on map.")
        return None

    location_counts = df_loc.groupby(['location', 'lat', 'lon']).size().reset_index(name='mentions')

    fig = px.scatter_mapbox(
        location_counts, lat="lat", lon="lon", size="mentions", color="mentions",
        hover_name="location", hover_data={"lat": False, "lon": False, "mentions": True},
        color_continuous_scale=px.colors.cyclical.IceFire, size_max=30, zoom=5,
        center={"lat": 23.6850, "lon": 90.3563},
        title="Geographic Hotspots for Prime Bank Mentions",
        mapbox_style="carto-positron"
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig