#!/usr/bin/env python3
"""
Dynamic Evidence Retriever
Fetches evidence from external sources for any input text, not just pre-detected claims
"""

import requests
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class DynamicEvidence:
    """Evidence retrieved from external sources."""
    id: str
    title: str
    content: str
    source: str
    url: str
    relevance_score: float
    source_type: str  # "wikipedia", "news", "factcheck", "scientific"
    timestamp: str

class DynamicEvidenceRetriever:
    """
    Retrieves evidence from multiple external sources for any input text.
    """
    
    def __init__(self):
        self.wikipedia_api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.news_api_key = None  # Set your News API key here
        self.factcheck_sources = [
            "https://www.snopes.com",
            "https://www.politifact.com",
            "https://www.factcheck.org"
        ]
        
    def retrieve_evidence_for_text(self, text: str, max_evidence: int = 5) -> List[DynamicEvidence]:
        """
        Retrieve evidence for any input text from multiple sources.
        
        Args:
            text: Input text to find evidence for
            max_evidence: Maximum number of evidence items to return
            
        Returns:
            List of evidence items from various sources
        """
        evidence = []
        
        # Extract key terms for search
        search_terms = self._extract_search_terms(text)
        
        # 1. Wikipedia evidence
        wiki_evidence = self._get_wikipedia_evidence(search_terms, max_evidence // 3)
        evidence.extend(wiki_evidence)
        
        # 2. News evidence
        news_evidence = self._get_news_evidence(search_terms, max_evidence // 3)
        evidence.extend(news_evidence)
        
        # 3. Fact-check evidence
        factcheck_evidence = self._get_factcheck_evidence(search_terms, max_evidence // 3)
        evidence.extend(factcheck_evidence)
        
        # Sort by relevance and return top results
        evidence.sort(key=lambda x: x.relevance_score, reverse=True)
        return evidence[:max_evidence]
    
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract key search terms from input text with enhanced relevance."""
        import re
        
        # Remove common words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'she', 'his', 'her', 'him', 'i', 'me', 'my'}
        
        # Extract words and phrases
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Add important phrases (2-3 word combinations)
        phrases = []
        words_list = text.lower().split()
        for i in range(len(words_list) - 1):
            phrase = f"{words_list[i]} {words_list[i+1]}"
            if len(phrase) > 5 and not any(stop in phrase for stop in stop_words):
                phrases.append(phrase)
        
        # Combine keywords and phrases, prioritize longer terms
        all_terms = keywords + phrases
        unique_terms = list(set(all_terms))
        unique_terms.sort(key=len, reverse=True)
        
        return unique_terms[:8]  # Top 8 terms for better coverage
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content."""
        import re
        
        query_terms = set(re.findall(r'\b[a-zA-Z]+\b', query.lower()))
        content_terms = set(re.findall(r'\b[a-zA-Z]+\b', content.lower()))
        
        if not query_terms:
            return 0.0
        
        # Calculate term overlap
        matching_terms = query_terms.intersection(content_terms)
        overlap_ratio = len(matching_terms) / len(query_terms)
        
        # Boost score for exact phrase matches
        phrase_boost = 0.0
        query_lower = query.lower()
        content_lower = content.lower()
        
        if query_lower in content_lower:
            phrase_boost = 0.3
        
        # Calculate final relevance score
        relevance = min(1.0, overlap_ratio + phrase_boost)
        
        return relevance
    
    def _get_wikipedia_evidence(self, search_terms: List[str], max_results: int) -> List[DynamicEvidence]:
        """Get evidence from Wikipedia API with enhanced search and content extraction."""
        evidence = []
        
        for term in search_terms[:2]:  # Use top 2 terms
            try:
                # Enhanced Wikipedia search with better query construction
                search_query = f"{term}"
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&format=json&srlimit=10&srnamespace=0"
                response = requests.get(search_url, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                if 'query' in data and 'search' in data['query']:
                    for result in data['query']['search']:
                        try:
                            # Get detailed page content
                            page_id = result['pageid']
                            page_title = result['title']
                            
                            # Get page summary and content
                            content_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&explaintext=true&pageids={page_id}&format=json"
                            content_response = requests.get(content_url, timeout=15)
                            
                            if content_response.status_code == 200:
                                content_data = content_response.json()
                                if 'query' in content_data and 'pages' in content_data['query']:
                                    page_data = content_data['query']['pages'].get(str(page_id), {})
                                    extract = page_data.get('extract', '')
                                    
                                    # Get page URL
                                    page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                                    
                                    # Calculate relevance based on content matching
                                    relevance = self._calculate_relevance(term, extract)
                                    
                                    if relevance > 0.2:  # Lower threshold for better coverage
                                        evidence.append(DynamicEvidence(
                                            id=f"wiki_{page_id}",
                                            title=page_title,
                                            content=extract[:800],  # First 800 chars
                                            source="Wikipedia",
                                            url=page_url,
                                            relevance_score=relevance,
                                            source_type="wikipedia",
                                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                                        ))
                                        
                                        if len(evidence) >= max_results:
                                            break
                        except Exception as e:
                            print(f"Error processing Wikipedia page {page_id}: {e}")
                            continue
                                
            except Exception as e:
                print(f"Error fetching Wikipedia evidence for '{term}': {e}")
                continue
                
        return evidence
    
    def _get_news_evidence(self, search_terms: List[str], max_results: int) -> List[DynamicEvidence]:
        """Get evidence from news sources."""
        evidence = []
        
        # Use NewsAPI if available, otherwise use a simple web search
        if self.news_api_key:
            evidence.extend(self._get_newsapi_evidence(search_terms, max_results))
        else:
            evidence.extend(self._get_web_search_evidence(search_terms, max_results))
            
        return evidence
    
    def _get_newsapi_evidence(self, search_terms: List[str], max_results: int) -> List[DynamicEvidence]:
        """Get evidence from NewsAPI."""
        evidence = []
        
        try:
            query = " ".join(search_terms[:3])
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&pageSize={max_results}&apiKey={self.news_api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'articles' in data:
                for article in data['articles']:
                    evidence.append(DynamicEvidence(
                        id=f"news_{hash(article.get('url', ''))}",
                        title=article.get('title', ''),
                        content=article.get('description', '')[:500],
                        source=article.get('source', {}).get('name', 'News'),
                        url=article.get('url', ''),
                        relevance_score=0.7,
                        source_type="news",
                        timestamp=article.get('publishedAt', '')
                    ))
                    
        except Exception as e:
            print(f"Error fetching news evidence: {e}")
            
        return evidence
    
    def _get_web_search_evidence(self, search_terms: List[str], max_results: int) -> List[DynamicEvidence]:
        """Get evidence using Google News search."""
        evidence = []
        
        query = " ".join(search_terms[:3])
        
        try:
            # Google News search
            news_search_url = f"https://news.google.com/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(news_search_url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for news articles
                articles = soup.find_all('article', class_='MQsxIb')[:max_results]
                
                for article in articles:
                    try:
                        title_elem = article.find('h3')
                        link_elem = article.find('a')
                        source_elem = article.find('time')
                        
                        if title_elem and link_elem:
                            title = title_elem.get_text(strip=True)
                            url = link_elem.get('href')
                            if url and not url.startswith('http'):
                                url = f"https://news.google.com{url}"
                            
                            source = "Google News"
                            if source_elem:
                                source = source_elem.get_text(strip=True)
                            
                            # Get article content if possible
                            try:
                                article_response = requests.get(url, timeout=10, headers={
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                })
                                
                                if article_response.status_code == 200:
                                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                                    # Try to find article content
                                    content_elem = article_soup.find('article') or article_soup.find('div', class_='content') or article_soup.find('p')
                                    content = content_elem.get_text(strip=True)[:600] if content_elem else title
                                else:
                                    content = title
                            except:
                                content = title
                            
                            relevance = self._calculate_relevance(query, content)
                            
                            if relevance > 0.3:
                                evidence.append(DynamicEvidence(
                                    id=f"news_{hash(title)}",
                                    title=title,
                                    content=content,
                                    source=source,
                                    url=url,
                                    relevance_score=relevance,
                                    source_type="news",
                                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                                ))
                    except Exception as e:
                        print(f"Error processing news article: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error fetching news evidence: {e}")
        
        return evidence
    
    def _get_factcheck_evidence(self, search_terms: List[str], max_results: int) -> List[DynamicEvidence]:
        """Get evidence from real fact-checking websites."""
        evidence = []
        
        query = " ".join(search_terms[:3])
        
        # Snopes API (if available) or web scraping
        try:
            # Try to get from Snopes search
            snopes_search_url = f"https://www.snopes.com/?s={query.replace(' ', '+')}"
            response = requests.get(snopes_search_url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                # Extract fact-check results from Snopes
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for fact-check articles
                articles = soup.find_all('article', class_='post')[:max_results]
                
                for article in articles:
                    try:
                        title_elem = article.find('h2', class_='post-title')
                        link_elem = article.find('a')
                        
                        if title_elem and link_elem:
                            title = title_elem.get_text(strip=True)
                            url = link_elem.get('href')
                            if url and not url.startswith('http'):
                                url = f"https://www.snopes.com{url}"
                            
                            # Get article content
                            article_response = requests.get(url, timeout=15, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            })
                            
                            if article_response.status_code == 200:
                                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                                content_elem = article_soup.find('div', class_='content')
                                
                                if content_elem:
                                    content = content_elem.get_text(strip=True)[:800]
                                    relevance = self._calculate_relevance(query, content)
                                    
                                    if relevance > 0.4:
                                        evidence.append(DynamicEvidence(
                                            id=f"snopes_{hash(title)}",
                                            title=title,
                                            content=content,
                                            source="Snopes",
                                            url=url,
                                            relevance_score=relevance,
                                            source_type="factcheck",
                                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                                        ))
                    except Exception as e:
                        print(f"Error processing Snopes article: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error fetching Snopes evidence: {e}")
        
        # Add PolitiFact evidence
        try:
            politifact_search_url = f"https://www.politifact.com/search/?q={query.replace(' ', '+')}"
            response = requests.get(politifact_search_url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for fact-check results
                articles = soup.find_all('div', class_='statement')[:max_results]
                
                for article in articles:
                    try:
                        title_elem = article.find('h3')
                        link_elem = article.find('a')
                        
                        if title_elem and link_elem:
                            title = title_elem.get_text(strip=True)
                            url = link_elem.get('href')
                            if url and not url.startswith('http'):
                                url = f"https://www.politifact.com{url}"
                            
                            # Get verdict
                            verdict_elem = article.find('div', class_='meter')
                            verdict = verdict_elem.get_text(strip=True) if verdict_elem else "Unknown"
                            
                            content = f"PolitiFact verdict: {verdict}. {title}"
                            relevance = self._calculate_relevance(query, content)
                            
                            if relevance > 0.4:
                                evidence.append(DynamicEvidence(
                                    id=f"politifact_{hash(title)}",
                                    title=f"PolitiFact: {title}",
                                    content=content,
                                    source="PolitiFact",
                                    url=url,
                                    relevance_score=relevance,
                                    source_type="factcheck",
                                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                                ))
                    except Exception as e:
                        print(f"Error processing PolitiFact article: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error fetching PolitiFact evidence: {e}")
        
        return evidence

# Convenience function for easy integration
def get_dynamic_evidence(text: str, max_evidence: int = 5) -> List[DynamicEvidence]:
    """Get dynamic evidence for any input text."""
    retriever = DynamicEvidenceRetriever()
    return retriever.retrieve_evidence_for_text(text, max_evidence)
