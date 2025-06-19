# src/insights_generator.py

import pandas as pd
from collections import Counter
import re

class InsightsGenerator:
    """
    Analyzes processed data to generate qualitative and quantitative insights
    in a structured dictionary format, ready for display in the dashboard.
    """
    def __init__(self):
        self.insights = {}
    
    def generate_all_insights(self, df, prime_df):
        """
        Orchestrator method to generate all insights for the dashboard.
        
        Args:
            df (pd.DataFrame): The full DataFrame with all posts.
            prime_df (pd.DataFrame): A filtered DataFrame with only Prime Bank posts.
            
        Returns:
            dict: A nested dictionary containing all generated insights.
        """
        total_posts = len(df)
        prime_posts = len(prime_df)
        prime_percentage = (prime_posts / total_posts * 100) if total_posts > 0 else 0
        
        self.insights['overview'] = {
            'summary': f"Analyzed {total_posts:,} total posts, of which {prime_posts:,} ({prime_percentage:.1f}%) specifically mention Prime Bank.",
            'context': f"The remaining {total_posts - prime_posts:,} posts mention other banks or general banking topics."
        }
        
        # Generate insights for each section using the Prime Bank data
        self.insights['sentiment'] = self._generate_sentiment_insights(prime_df)
        self.insights['emotion'] = self._generate_emotion_insights(prime_df)
        self.insights['category'] = self._generate_category_insights(prime_df)
        self.insights['topics'] = self._generate_topic_insights(prime_df)
        self.insights['actions'] = self._generate_action_insights(prime_df)
        
        # Comparative analysis uses the full dataset
        self.insights['comparison'] = self._generate_comparison_insights(df)
        
        return self.insights
    
    def _generate_sentiment_insights(self, df):
        """Generate sentiment-specific insights with examples and context."""
        if len(df) == 0:
            return {
                'summary': 'No Prime Bank posts found for sentiment analysis.',
                'positive_context': '', 'negative_context': '', 'neutral_context': '',
                'examples': {'Positive': [], 'Negative': [], 'Neutral': []}
            }
        
        sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
        
        # Get up to 3 sample posts for each sentiment
        sentiment_examples = {
            s: df[df['sentiment'] == s]['text'].head(3).tolist() 
            for s in ['Positive', 'Negative', 'Neutral']
        }
        
        # Analyze themes in negative posts
        negative_posts_text = ' '.join(df[df['sentiment'] == 'Negative']['text'].astype(str).tolist()).lower()
        negative_themes = [theme for theme in ['wait times', 'fees', 'charges', 'app issue', 'online banking', 'customer service'] if theme in negative_posts_text]

        # Analyze themes in neutral posts (typically inquiries)
        neutral_posts_text = ' '.join(df[df['sentiment'] == 'Neutral']['text'].astype(str).tolist()).lower()
        neutral_themes = [theme for theme in ['how to', 'what is', 'interest rate', 'loan', 'account', 'credit card'] if theme in neutral_posts_text]
        
        return {
            'summary': f"Sentiment Breakdown: {sentiment_dist.get('Positive', 0):.1f}% Positive, {sentiment_dist.get('Negative', 0):.1f}% Negative, {sentiment_dist.get('Neutral', 0):.1f}% Neutral.",
            'positive_context': "Positive posts often praise specific aspects like 'customer service' and the 'mobile app'.",
            'negative_context': f"Negative posts frequently mention issues like {', '.join(negative_themes) if negative_themes else 'service delays and technical problems'}.",
            'neutral_context': f"Neutral posts are primarily informational, often asking about {', '.join(neutral_themes) if neutral_themes else 'general services and account details'}.",
            'examples': sentiment_examples
        }

    def _generate_emotion_insights(self, df):
        """Generate emotion-specific insights with examples."""
        if len(df) == 0 or 'emotion' not in df.columns or df['emotion'].value_counts().empty:
            return {
                'summary': 'No Prime Bank posts found for emotion analysis.',
                'distribution': {}, 'examples': {}, 'recommendation': ''
            }
            
        emotion_dist = df['emotion'].value_counts()
        top_emotion = emotion_dist.index[0]

        # Get up to 2 examples for each of the top 4 emotions
        emotion_examples = {
            emotion: df[df['emotion'] == emotion]['text'].head(2).tolist()
            for emotion in emotion_dist.head(4).index if emotion != 'Neutral'
        }
        
        return {
            'summary': f"The dominant emotion expressed is '{top_emotion}' with {emotion_dist.iloc[0]} mentions.",
            'distribution': emotion_dist.to_dict(),
            'examples': emotion_examples,
            'recommendation': self._get_emotion_recommendation(emotion_dist)
        }

    def _generate_category_insights(self, df):
        """Generate category-specific insights."""
        if len(df) == 0:
            return {'summary': 'No posts found for category analysis.'}
        
        category_dist = df['category'].value_counts()
        return {
            'summary': f"Post categories: {', '.join([f'{cat} ({count})' for cat, count in category_dist.items()])}",
            'distribution': category_dist.to_dict(),
            'urgent_attention': f"{category_dist.get('Complaint', 0)} complaints require attention.",
            'opportunities': f"{category_dist.get('Suggestion', 0)} suggestions offer improvement ideas."
        }

    def _generate_topic_insights(self, df):
        """Identify trending topics by counting keyword occurrences."""
        if len(df) == 0:
            return {'summary': 'No posts found for topic analysis.'}
        
        all_text = ' '.join(df['text'].astype(str).tolist()).lower()
        
        topics = {
            'Digital Banking': ['app', 'online', 'mobile', 'website', 'internet banking', 'crashing'],
            'Customer Service': ['staff', 'service', 'help', 'support', 'employee', 'behavior'],
            'Fees & Charges': ['fee', 'charge', 'cost', 'expensive', 'hidden'],
            'Loans & Credit': ['loan', 'credit', 'mortgage', 'interest', 'emi', 'card'],
            'Branch & ATM': ['atm', 'branch', 'location', 'machine', 'cash', 'queue', 'wait']
        }
        
        # Count total occurrences of all keywords for a topic
        topic_counts = {topic: sum(all_text.count(kw) for kw in keywords) for topic, keywords in topics.items()}
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        trending_topic = sorted_topics[0][0] if sorted_topics and sorted_topics[0][1] > 0 else 'None'

        return {
            'summary': f"Top discussed topics are: {', '.join([t[0] for t in sorted_topics[:3]])}.",
            'all_topics': dict(sorted_topics),
            'trending': trending_topic,
            'recommendation': f"Focus on improving {trending_topic.lower()} due to high discussion volume." if trending_topic != 'None' else "No clear topic trends."
        }

    def _generate_comparison_insights(self, df):
        """Compare Prime Bank's sentiment with other mentioned banks."""
        bank_sentiment = {}
        all_banks = df['primary_bank'].unique()
        
        for bank in all_banks:
            if bank == 'none' or bank == 'multiple':
                continue
            
            bank_posts = df[df['primary_bank'] == bank]
            if not bank_posts.empty:
                positive_rate = (bank_posts['sentiment'] == 'Positive').sum() / len(bank_posts) * 100
                bank_sentiment[bank] = {
                    'posts': len(bank_posts),
                    'positive_rate': positive_rate
                }
        
        if 'prime_bank' not in bank_sentiment:
            return {'summary': 'No Prime Bank posts found for comparison.'}
        
        prime_positive = bank_sentiment['prime_bank']['positive_rate']
        avg_competitor_rate = pd.Series({b: d['positive_rate'] for b, d in bank_sentiment.items() if b != 'prime_bank'}).mean()
        
        comparison = "above" if prime_positive > avg_competitor_rate else "below"
        
        return {
            'summary': f"Prime Bank's positive sentiment ({prime_positive:.1f}%) is {comparison} the competitor average of {avg_competitor_rate:.1f}%.",
            'comparison': bank_sentiment,
            'recommendation': "Focus on maintaining positive momentum." if comparison == 'above' else "Urgent improvement needed to match competitor satisfaction."
        }

    def _generate_action_insights(self, df):
        """Generate actionable insights based on high-priority posts."""
        if len(df) == 0:
            return {'summary': 'No posts found for action analysis.'}
        
        high_priority = df[
            (df['sentiment'] == 'Negative') & 
            (df['category'] == 'Complaint')
        ]
        
        quick_wins = df[df['sentiment'] == 'Positive'].nlargest(5, 'viral_score')
        
        return {
            'immediate': {
                'count': len(high_priority),
                'description': 'High-priority complaints (Negative sentiment + Complaint category) require immediate response.',
                'action': 'Review these posts and contact customers within 24 hours.'
            },
            'quick_wins': {
                'count': len(quick_wins),
                'description': 'Positive testimonials with high viral scores are available for marketing.',
                'action': 'Amplify these success stories and thank the customers publicly.'
            }
        }

    def _get_emotion_recommendation(self, emotion_dist):
        """Get a recommendation based on the dominant emotion."""
        if emotion_dist.empty:
            return "No emotional data to analyze."
        
        top_emotion = emotion_dist.index[0]
        
        recommendations = {
            'Joy': "Leverage positive emotions by encouraging happy customers to share testimonials.",
            'Frustration': "Implement a rapid response protocol for frustrated customers to prevent escalation.",
            'Confusion': "Create clearer communication materials and improve the FAQ/help section.",
            'Anxiety': "Provide reassurance through proactive communication about security and processes.",
            'Neutral': "Engage neutral customers with targeted campaigns to foster a positive connection."
        }
        return recommendations.get(top_emotion, "Monitor customer emotions closely.")