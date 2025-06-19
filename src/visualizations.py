import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_sentiment_pie(df):
    """Create sentiment distribution pie chart"""
    sentiment_counts = df['sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#95a5a6'
        }
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    return fig

def create_emotion_bar(df):
    """Create emotion distribution bar chart"""
    emotion_counts = df['emotion'].value_counts()
    
    # Define colors for emotions
    color_map = {
        'Joy': '#f39c12',
        'Frustration': '#e74c3c',
        'Confusion': '#3498db',
        'Anxiety': '#9b59b6',
        'Neutral': '#95a5a6'
    }
    
    colors = [color_map.get(emotion, '#95a5a6') for emotion in emotion_counts.index]
    
    fig = px.bar(
        x=emotion_counts.index,
        y=emotion_counts.values,
        title="Emotion Detection",
        labels={'x': 'Emotion', 'y': 'Count'},
        color=emotion_counts.index,
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )
    
    return fig

def create_category_donut(df):
    """Create post category donut chart"""
    category_counts = df['category'].value_counts()
    
    # Define colors for categories
    color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Post Categories",
        hole=0.4,
        color_discrete_sequence=color_sequence
    )
    
    # Add text in center
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    # Add annotation in center
    fig.add_annotation(
        text=f"Total<br>{len(df)}",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=20)
    )
    
    return fig

def create_mentions_timeline(df):
    """Create timeline of Prime Bank mentions if date column exists"""
    date_columns = ['date', 'created_at', 'timestamp', 'Date', 'post_date']
    date_col = None
    
    # Find date column
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    if not date_col:
        return None
    
    try:
        # Parse dates
        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Remove invalid dates
        df_valid = df[df['date_parsed'].notna()]
        
        if len(df_valid) == 0:
            return None
        
        # Group by date
        timeline_df = df_valid.groupby(df_valid['date_parsed'].dt.date).agg({
            'prime_mentions': 'sum',
            'sentiment': lambda x: (x == 'Positive').sum()
        }).reset_index()
        
        timeline_df.columns = ['date', 'mentions', 'positive_posts']
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add mentions line
        fig.add_trace(
            go.Scatter(
                x=timeline_df['date'],
                y=timeline_df['mentions'],
                name='Total Mentions',
                line=dict(color='#3498db', width=3),
                mode='lines+markers'
            ),
            secondary_y=False,
        )
        
        # Add positive posts line
        fig.add_trace(
            go.Scatter(
                x=timeline_df['date'],
                y=timeline_df['positive_posts'],
                name='Positive Posts',
                line=dict(color='#2ecc71', width=2, dash='dot'),
                mode='lines+markers'
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Number of Mentions", secondary_y=False)
        fig.update_yaxes(title_text="Positive Posts", secondary_y=True)
        
        fig.update_layout(
            title="Prime Bank Mentions Over Time",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating timeline: {e}")
        return None

def create_summary_metrics(df):
    """Calculate summary metrics for display"""
    total_posts = len(df)
    prime_posts = len(df[df['prime_mentions'] > 0])
    total_mentions = df['prime_mentions'].sum()
    
    # Calculate positive sentiment rate
    if prime_posts > 0:
        prime_df = df[df['prime_mentions'] > 0]
        positive_rate = (prime_df['sentiment'] == 'Positive').sum() / prime_posts * 100
    else:
        positive_rate = 0
    
    metrics = {
        'Total Posts Analyzed': f"{total_posts:,}",
        'Posts Mentioning Prime Bank': f"{prime_posts:,}",
        'Total Prime Bank Mentions': f"{total_mentions:,}",
        'Positive Sentiment Rate': f"{positive_rate:.1f}%"
    }
    
    return metrics

def create_viral_posts_chart(df, top_n=10):
    """Create horizontal bar chart of most viral posts"""
    # Get top viral posts
    top_viral = df.nlargest(top_n, 'viral_score')
    
    # Truncate text for display
    top_viral['text_truncated'] = top_viral['text'].apply(
        lambda x: x[:50] + '...' if len(str(x)) > 50 else x
    )
    
    # Create horizontal bar chart
    fig = px.bar(
        top_viral,
        x='viral_score',
        y='text_truncated',
        orientation='h',
        title=f'Top {top_n} Viral Posts',
        color='sentiment',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#95a5a6'
        },
        hover_data=['text', 'emotion', 'category']
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Viral Score",
        yaxis_title="Post Preview",
        showlegend=True
    )
    
    return fig

def create_word_frequency_chart(df, top_n=15):
    """Create word frequency chart for Prime Bank posts"""
    from collections import Counter
    import re
    
    # Get only Prime Bank posts
    prime_posts = df[df['prime_mentions'] > 0]['text'].dropna()
    
    if len(prime_posts) == 0:
        return None
    
    # Combine all text
    all_text = ' '.join(prime_posts.astype(str)).lower()
    
    # Remove common words and Prime Bank itself
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'prime', 'bank', 'primebank', 'i', 'me', 'my',
        'we', 'you', 'your', 'they', 'their', 'this', 'that', 'these', 'those'
    }
    
    # Extract words
    words = re.findall(r'\b[a-z]+\b', all_text)
    words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count frequency
    word_freq = Counter(words).most_common(top_n)
    
    if not word_freq:
        return None
    
    # Create dataframe
    freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    
    # Create bar chart
    fig = px.bar(
        freq_df,
        x='Frequency',
        y='Word',
        orientation='h',
        title=f'Top {top_n} Words in Prime Bank Posts',
        color='Frequency',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

def create_sentiment_by_category(df):
    """Create stacked bar chart of sentiment by category"""
    # Filter for Prime Bank mentions
    prime_df = df[df['prime_mentions'] > 0]
    
    if len(prime_df) == 0:
        return None
    
    # Create crosstab
    sentiment_category = pd.crosstab(
        prime_df['category'],
        prime_df['sentiment'],
        normalize='index'
    ) * 100
    
    # Create stacked bar chart
    fig = go.Figure()
    
    sentiments = ['Positive', 'Negative', 'Neutral']
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
    
    for sentiment in sentiments:
        if sentiment in sentiment_category.columns:
            fig.add_trace(go.Bar(
                name=sentiment,
                x=sentiment_category.index,
                y=sentiment_category[sentiment],
                marker_color=colors.get(sentiment, '#95a5a6'),
                hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
            ))
    
    fig.update_layout(
        barmode='stack',
        title='Sentiment Distribution by Post Category',
        xaxis_title='Category',
        yaxis_title='Percentage',
        yaxis=dict(tickformat='.0f', ticksuffix='%'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_priority_matrix(df):
    """Create scatter plot showing priority posts"""
    # Filter for Prime Bank mentions
    prime_df = df[df['prime_mentions'] > 0].copy()
    
    if len(prime_df) == 0:
        return None
    
    # Calculate urgency score (based on negative sentiment + complaints)
    prime_df['urgency'] = 0
    prime_df.loc[prime_df['sentiment'] == 'Negative', 'urgency'] += 2
    prime_df.loc[prime_df['category'] == 'Complaint', 'urgency'] += 2
    prime_df.loc[prime_df['emotion'].isin(['Frustration', 'Anxiety']), 'urgency'] += 1
    
    # Create scatter plot
    fig = px.scatter(
        prime_df,
        x='viral_score',
        y='urgency',
        size='prime_mentions',
        color='sentiment',
        hover_data=['text', 'emotion', 'category'],
        title='Priority Matrix: Viral Score vs Urgency',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#95a5a6'
        }
    )
    
    # Add quadrant lines
    fig.add_hline(y=2.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=prime_df['viral_score'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=0.95, y=0.95, text="High Priority", xref="paper", yref="paper", showarrow=False)
    fig.add_annotation(x=0.05, y=0.95, text="Monitor", xref="paper", yref="paper", showarrow=False)
    
    fig.update_layout(
        xaxis_title="Viral Score (Reach)",
        yaxis_title="Urgency Score",
        showlegend=True
    )
    
    return fig