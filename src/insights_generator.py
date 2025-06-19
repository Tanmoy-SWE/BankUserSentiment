# src/insights_generator.py

import pandas as pd
from collections import Counter
import re

class InsightsGenerator:
    """
    Analyzes processed data to generate rich, qualitative insights for the dashboard.
    """
    def __init__(self):
        self.insights = {}
    
    def generate_all_insights(self, posts_df, all_text_df):
        """
        Orchestrator to generate all insights based on the specified dataframes.
        """
        # --- THE DEFINITIVE FIX ---
        # Before doing anything, ensure the required column exists. If not, create empty frames.
        if 'prime_mentions' not in posts_df.columns:
            prime_posts_df = pd.DataFrame()
        else:
            prime_posts_df = posts_df[posts_df['prime_mentions'] > 0]

        if 'prime_mentions' not in all_text_df.columns:
            prime_all_text_df = pd.DataFrame()
        else:
            prime_all_text_df = all_text_df[all_text_df['prime_mentions'] > 0]
        # --- END OF FIX ---

        # Requirement 3: Sentiment (Posts Only)
        self.insights['sentiment'] = self._generate_sentiment_insights(prime_posts_df)
        
        # Requirement 4: Emotion (Posts & Comments)
        self.insights['emotion'] = self._generate_emotion_insights(prime_all_text_df)
        
        # Requirement 2: Categories (Posts & Comments)
        self.insights['category'] = self._generate_category_insights(prime_all_text_df)
        
        return self.insights
    
    def _get_common_words(self, text_series, top_n=5):
        """Helper function to find common keywords in a series of text."""
        if text_series.empty:
            return "No data"
        
        stop_words = {'the', 'a', 'an', 'is', 'i', 'to', 'for', 'in', 'it', 'and', 'my', 'of', 'prime', 'bank'}
        all_text = ' '.join(text_series.astype(str).tolist()).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_text) # Find words with 4+ letters
        filtered_words = [word for word in words if word not in stop_words]
        
        if not filtered_words:
            return "general topics"
            
        return ', '.join([word for word, count in Counter(filtered_words).most_common(top_n)])

    def _generate_sentiment_insights(self, df):
        """Analyzes sentiment on POSTS ONLY and provides detailed summaries."""
        if df.empty:
            return {
                'summary': 'No posts found for sentiment analysis.',
                'positive_themes': 'N/A',
                'negative_themes': 'N/A',
                'negative_examples': []
            }
        
        dist = df['sentiment'].value_counts(normalize=True) * 100
        
        # Analyze themes within each sentiment
        positive_df = df[df['sentiment'] == 'Positive']
        negative_df = df[df['sentiment'] == 'Negative']
        
        positive_themes = self._get_common_words(positive_df['text'])
        negative_themes = self._get_common_words(negative_df['text'])
        
        return {
            'summary': f"Positive: {dist.get('Positive', 0):.1f}%, Negative: {dist.get('Negative', 0):.1f}%, Neutral: {dist.get('Neutral', 0):.1f}%",
            'positive_themes': f"Customers are happy about: {positive_themes}.",
            'negative_themes': f"Customers are unhappy about: {negative_themes}.",
            'negative_examples': negative_df['text'].head(3).tolist()
        }

    def _generate_emotion_insights(self, df):
        """Analyzes emotions on POSTS & COMMENTS and provides detailed summaries."""
        if df.empty:
            return {
                'summary': 'No text found for emotion analysis.',
                'details': {}
            }
        
        emotion_dist = df['emotion'].value_counts()
        top_emotion = emotion_dist.index[0] if not emotion_dist.empty else "N/A"
        
        insight_details = {}
        for emotion in ['Joy', 'Frustration', 'Confusion', 'Anxiety']:
            if emotion in df['emotion'].values:
                emotion_df = df[df['emotion'] == emotion]
                insight_details[emotion] = {
                    'themes': self._get_common_words(emotion_df['text'], 3),
                    'example': emotion_df['text'].iloc[0] if not emotion_df.empty else "N/A"
                }

        return {
            'summary': f"The most common emotion is '{top_emotion}'.",
            'details': insight_details
        }

    def _generate_category_insights(self, df):
        """Analyzes categories on POSTS & COMMENTS and provides detailed summaries."""
        if df.empty:
            return {
                'summary': 'No text found for category analysis.',
                'details': {}
            }

        insight_details = {}
        for category in ['Complaint', 'Inquiry', 'Praise', 'Suggestion']:
            if category in df['category'].values:
                category_df = df[df['category'] == category]
                insight_details[category] = {
                    'themes': self._get_common_words(category_df['text'], 4)
                }

        return {
            'summary': f"The most frequent category is '{df['category'].mode()[0]}'. Complaints and Inquiries are key areas to watch.",
            'details': insight_details
        }