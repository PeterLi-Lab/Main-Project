import os
import requests
import json
from typing import List, Dict, Optional

class ChatGPTAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize ChatGPT API analyzer"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_post_sentiment(self, title: str, body: str = "") -> Dict:
        """Analyze post sentiment and extract key insights"""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        prompt = f"""
        Analyze this post and provide insights in JSON format:
        
        Title: {title}
        Body: {body[:500]}...
        
        Please provide:
        1. Sentiment (positive/negative/neutral)
        2. Topic category (programming/error/question/tutorial/other)
        3. Complexity level (beginner/intermediate/advanced)
        4. Key tags (max 5)
        5. Engagement prediction (low/medium/high)
        6. Summary (max 100 words)
        
        Return as JSON with these exact keys: sentiment, topic, complexity, tags, engagement_prediction, summary
        """
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                # Try to parse JSON from response
                try:
                    return json.loads(content)
                except:
                    return {"raw_response": content}
            else:
                return {"error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def generate_smart_tags(self, title: str, body: str = "") -> List[str]:
        """Generate smart tags for posts"""
        if not self.api_key:
            return []
        
        prompt = f"""
        Generate 3-5 relevant tags for this post. Return only the tags separated by commas, no other text:
        
        Title: {title}
        Body: {body[:300]}...
        """
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.2
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return [tag.strip() for tag in content.split(',') if tag.strip()]
            else:
                return []
                
        except Exception as e:
            return []
    
    def analyze_post_quality(self, title: str, body: str = "") -> Dict:
        """Analyze post quality and provide improvement suggestions"""
        if not self.api_key:
            return {"error": "API key not configured"}
        
        prompt = f"""
        Analyze the quality of this post and provide suggestions:
        
        Title: {title}
        Body: {body[:800]}...
        
        Provide analysis in JSON format with:
        1. Quality_score (1-10)
        2. Clarity_score (1-10)
        3. Completeness_score (1-10)
        4. Suggestions (list of improvement tips)
        5. Best_practices (list of what was done well)
        
        Return as JSON with these exact keys: quality_score, clarity_score, completeness_score, suggestions, best_practices
        """
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                try:
                    return json.loads(content)
                except:
                    return {"raw_response": content}
            else:
                return {"error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

def rule_based_analysis(title: str, body: str, post_length: int, num_tags: int, vote_ratio: float, total_votes: int, title_length: int) -> Dict:
    """Rule-based analysis as fallback when API is not available"""
    title_lower = title.lower()
    body_lower = str(body).lower()
    
    # Rule-based sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'awesome', 'helpful', 'solved', 'working', 'success']
    negative_words = ['error', 'problem', 'fail', 'crash', 'bug', 'issue', 'broken', 'wrong']
    
    positive_count = sum(1 for word in positive_words if word in title_lower or word in body_lower)
    negative_count = sum(1 for word in negative_words if word in title_lower or word in body_lower)
    
    if positive_count > negative_count:
        sentiment = 'positive'
    elif negative_count > positive_count:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    # Rule-based topic classification
    if any(word in title_lower for word in ['error', 'exception', 'bug', 'fail']):
        topic = 'error_debugging'
    elif any(word in title_lower for word in ['how', 'what', 'why', 'when', 'where']):
        topic = 'question'
    elif any(word in title_lower for word in ['guide', 'tutorial', 'example']):
        topic = 'tutorial'
    elif any(word in title_lower for word in ['code', 'function', 'class', 'programming']):
        topic = 'programming'
    else:
        topic = 'general'
    
    # Rule-based complexity
    if post_length < 50:
        complexity = 'beginner'
    elif post_length > 200:
        complexity = 'advanced'
    else:
        complexity = 'intermediate'
    
    # Rule-based engagement prediction
    if total_votes > 10:
        engagement_prediction = 'high'
    elif total_votes < 2:
        engagement_prediction = 'low'
    else:
        engagement_prediction = 'medium'
    
    # Rule-based quality score
    quality_score = 5.0
    if post_length > 100:
        quality_score += 1
    if num_tags > 0:
        quality_score += 1
    if vote_ratio > 0.7:
        quality_score += 1
    if title_length > 5:
        quality_score += 1
    
    quality_score = min(10.0, quality_score)
    
    return {
        'sentiment': sentiment,
        'topic': topic,
        'complexity': complexity,
        'engagement_prediction': engagement_prediction,
        'quality_score': quality_score
    } 