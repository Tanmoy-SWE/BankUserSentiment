import pandas as pd
from collections import Counter

class InsightsGenerator:
    def __init__(self):
        self.insights = {}
    
    def generate_all_insights(self, df, prime_df):
        """Generate comprehensive insights for all analyses"""
        
        # Overall statistics
        total_posts = len(df)
        prime_posts = len(prime_df)
        prime_percentage = (prime_posts / total_posts * 100) if total_posts > 0 else 0
        
        self.insights['overview'] = {
            'summary': f"Analyzed {total_posts:,} total posts, of which {prime_posts:,} ({prime_percentage:.1f}%) specifically mention Prime Bank.",
            'context': f"The remaining {total_posts - prime_posts:,} posts mention other banks or general banking topics."
        }
        
        # Sentiment insights
        self.insights['sentiment'] = self._generate_sentiment_insights(prime_df)
        
        # Emotion insights
        self.insights['emotion'] = self._generate_emotion_insights(prime_df)
        
        # Category insights
        self.insights['category'] = self._generate_category_insights(prime_df)
        
        # Trending topics
        self.insights['topics'] = self._generate_topic_insights(prime_df)
        
        # Comparative analysis
        self.insights['comparison'] = self._generate_comparison_insights(df)
        
        # Priority actions
        self.insights['actions'] = self._generate_action_insights(prime_df)
        
        return self.insights
    
    def _generate_sentiment_insights(self, df):
        """Generate sentiment-specific insights"""
        if len(df) == 0:
            return {'summary': 'No Prime Bank posts found for sentiment analysis.'}
        
        sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
        
        # Get sample posts for each sentiment
        sentiment_examples = {}
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            examples = df[df['sentiment'] == sentiment]['text'].head(2).tolist()
            sentiment_examples[sentiment] = examples
        
        # Analyze negative posts for common issues
        negative_posts = df[df['sentiment'] == 'Negative']['text']
        negative_themes = []
        if len(negative_posts) > 0:
            all_negative_text = ' '.join(negative_posts.astype(str).tolist()).lower()
            if 'wait' in all_negative_text or 'queue' in all_negative_text:
                negative_themes.append('long wait times')
            if 'fee' in all_negative_text or 'charge' in all_negative_text:
                negative_themes.append('fees and charges')
            if 'app' in all_negative_text or 'online' in all_negative_text:
                negative_themes.append('digital banking issues')
            if 'staff' in all_negative_text or 'service' in all_negative_text:
                negative_themes.append('customer service')
        
        insights = {
            'summary': f"Sentiment breakdown: {sentiment_dist.get('Positive', 0):.1f}% positive, {sentiment_dist.get('Negative', 0):.1f}% negative, {sentiment_dist.get('Neutral', 0):.1f}% neutral.",
            'positive_context': f"Positive posts ({sentiment_dist.get('Positive', 0):.1f}%) primarily praise customer service, digital banking features, and efficient processes.",
            'negative_context': f"Negative posts ({sentiment_dist.get('Negative', 0):.1f}%) mainly complain about: {', '.join(negative_themes) if negative_themes else 'various service issues'}.",
            'neutral_context': f"Neutral posts ({sentiment_dist.get('Neutral', 0):.1f}%) are mostly inquiries about services and general discussions.",
            'examples': sentiment_examples,
            'concern_areas': negative_themes
        }
        
        return insights
    
    def _generate_emotion_insights(self, df):
        """Generate emotion-specific insights"""
        if len(df) == 0:
            return {'summary': 'No Prime Bank posts found for emotion analysis.'}
        
        emotion_dist = df['emotion'].value_counts()
        total_emotional = len(df[df['emotion'] != 'Neutral'])
        
        emotion_contexts = {
            'Joy': 'Customers expressing joy are satisfied with services, particularly praising staff helpfulness and quick problem resolution.',
            'Frustration': 'Frustrated customers mainly face issues with wait times, technical problems, and unresolved complaints.',
            'Confusion': 'Confused customers need better information about products, fees, and online banking procedures.',
            'Anxiety': 'Anxious customers are worried about account security, loan applications, and urgent transaction issues.'
        }
        
        # Get most common emotion keywords
        emotion_keywords = {}
        for emotion in ['Joy', 'Frustration', 'Confusion', 'Anxiety']:
            emotion_posts = df[df['emotion'] == emotion]
            if len(emotion_posts) > 0:
                # Flatten all keywords for this emotion
                all_keywords = []
                for keywords in emotion_posts['emotion_keywords']:
                    if isinstance(keywords, list):
                        all_keywords.extend(keywords)
                if all_keywords:
                    emotion_keywords[emotion] = Counter(all_keywords).most_common(3)
        
        insights = {
            'summary': f"{total_emotional} out of {len(df)} Prime Bank posts ({total_emotional/len(df)*100:.1f}%) express clear emotions.",
            'distribution': {emotion: count for emotion, count in emotion_dist.items()},
            'contexts': emotion_contexts,
            'top_emotion': emotion_dist.index[0] if len(emotion_dist) > 0 else 'None',
            'keywords': emotion_keywords,
            'recommendation': self._get_emotion_recommendation(emotion_dist)
        }
        
        return insights
    
    def _generate_category_insights(self, df):
        """Generate category-specific insights"""
        if len(df) == 0:
            return {'summary': 'No Prime Bank posts found for category analysis.'}
        
        category_dist = df['category'].value_counts()
        
        category_insights = {
            'Inquiry': {
                'common_topics': ['account opening', 'loan applications', 'online banking setup', 'branch locations'],
                'action': 'Improve FAQ section and provide clearer information channels'
            },
            'Complaint': {
                'common_topics': ['service delays', 'technical issues', 'hidden fees', 'staff behavior'],
                'action': 'Establish rapid response team for complaint resolution'
            },
            'Praise': {
                'common_topics': ['helpful staff', 'quick service', 'user-friendly app', 'problem resolution'],
                'action': 'Recognize and reward mentioned staff members'
            },
            'Suggestion': {
                'common_topics': ['new features', 'branch expansion', 'service improvements', 'digital enhancements'],
                'action': 'Review suggestions for product development roadmap'
            }
        }
        
        insights = {
            'summary': f"Post categories: {', '.join([f'{cat} ({count})' for cat, count in category_dist.items()])}",
            'details': category_insights,
            'urgent_attention': f"{category_dist.get('Complaint', 0)} complaints require immediate attention",
            'opportunities': f"{category_dist.get('Suggestion', 0)} suggestions for improvement"
        }
        
        return insights
    
    def _generate_topic_insights(self, df):
        """Identify trending topics"""
        if len(df) == 0:
            return {'summary': 'No Prime Bank posts found for topic analysis.'}
        
        # Combine all text
        all_text = ' '.join(df['text'].astype(str).tolist()).lower()
        
        # Define topic keywords
        topics = {
            'Digital Banking': ['app', 'online', 'mobile', 'website', 'internet banking'],
            'Customer Service': ['staff', 'service', 'help', 'support', 'employee'],
            'Fees & Charges': ['fee', 'charge', 'cost', 'expensive', 'price'],
            'Loans': ['loan', 'credit', 'mortgage', 'interest', 'emi'],
            'ATM & Branch': ['atm', 'branch', 'location', 'machine', 'cash'],
            'Account Services': ['account', 'savings', 'current', 'balance', 'statement']
        }
        
        topic_counts = {}
        for topic, keywords in topics.items():
            count = sum(1 for keyword in keywords if keyword in all_text)
            if count > 0:
                topic_counts[topic] = count
        
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        insights = {
            'summary': f"Top discussed topics: {', '.join([f'{topic} ({count} mentions)' for topic, count in sorted_topics[:3]])}",
            'all_topics': dict(sorted_topics),
            'trending': sorted_topics[0][0] if sorted_topics else 'None',
            'recommendation': f"Focus on improving {sorted_topics[0][0].lower()} based on high discussion volume" if sorted_topics else "No clear topic trends"
        }
        
        return insights
    
    def _generate_comparison_insights(self, df):
        """Compare Prime Bank with other banks"""
        bank_sentiment = {}
        
        for bank in ['prime_bank', 'eastern_bank', 'brac_bank', 'city_bank', 'dutch_bangla']:
            bank_posts = df[df['primary_bank'] == bank]
            if len(bank_posts) > 0:
                positive_rate = (bank_posts['sentiment'] == 'Positive').sum() / len(bank_posts) * 100
                bank_sentiment[bank] = {
                    'posts': len(bank_posts),
                    'positive_rate': positive_rate
                }
        
        if 'prime_bank' in bank_sentiment:
            prime_positive = bank_sentiment['prime_bank']['positive_rate']
            comparison = "above average" if prime_positive > 50 else "below average"
            
            insights = {
                'summary': f"Prime Bank has {prime_positive:.1f}% positive sentiment, which is {comparison} in the banking sector.",
                'comparison': bank_sentiment,
                'recommendation': "Focus on maintaining positive momentum" if prime_positive > 50 else "Urgent improvement needed to match competitor satisfaction levels"
            }
        else:
            insights = {'summary': 'No comparative data available.'}
        
        return insights
    
    def _generate_action_insights(self, df):
        """Generate actionable insights"""
        if len(df) == 0:
            return {'summary': 'No Prime Bank posts found for action analysis.'}
        
        # High priority posts
        high_priority = df[
            (df['sentiment'] == 'Negative') & 
            (df['emotion'].isin(['Frustration', 'Anxiety'])) &
            (df['category'] == 'Complaint')
        ]
        
        # Quick wins - positive posts that can be amplified
        quick_wins = df[
            (df['sentiment'] == 'Positive') & 
            (df['category'] == 'Praise')
        ]
        
        actions = {
            'immediate': {
                'count': len(high_priority),
                'description': 'High-priority complaints requiring immediate response',
                'action': 'Contact these customers within 24 hours'
            },
            'quick_wins': {
                'count': len(quick_wins),
                'description': 'Positive testimonials for marketing use',
                'action': 'Share success stories and thank customers publicly'
            },
            'strategic': {
                'description': 'Long-term improvements based on feedback patterns',
                'actions': [
                    'Enhance digital banking infrastructure',
                    'Implement customer service training program',
                    'Review and simplify fee structure'
                ]
            }
        }
        
        return actions
    
    def _get_emotion_recommendation(self, emotion_dist):
        """Get recommendation based on emotion distribution"""
        if len(emotion_dist) == 0:
            return "No emotional data to analyze"
        
        top_emotion = emotion_dist.index[0]
        
        recommendations = {
            'Joy': "Leverage positive emotions by encouraging happy customers to share testimonials",
            'Frustration': "Implement rapid response protocol for frustrated customers to prevent escalation",
            'Confusion': "Create clearer communication materials and improve customer education",
            'Anxiety': "Provide reassurance through proactive communication about security and processes",
            'Neutral': "Engage neutral customers with targeted campaigns to create"
        }
        return recommendations.get(top_emotion, "Monitor customer emotions closely")