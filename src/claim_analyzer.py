#!/usr/bin/env python3
"""
Enhanced Claim Analyzer for TruthLens
Uses News API with semantic search, improved stance detection, and better verdict aggregation.
"""

import re
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# Import required components
try:
    from src.news.news_handler import NewsHandler, NewsArticle
    from src.verification.google_factcheck_api import GoogleFactCheckAPI, FactCheckResult
    from src.verification.enhanced_verifier import EnhancedVerifier
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some components not available: {e}")
    COMPONENTS_AVAILABLE = False

# Try to import sentence transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    logging.warning("Sentence transformers not available, falling back to keyword search")
    SEMANTIC_SEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceBadge:
    """Confidence badge with color coding."""
    level: str  # "Likely True", "Unclear", "Likely False", "Not Enough Info"
    confidence: float
    color: str  # "green", "yellow", "red", "gray"
    emoji: str  # "🟢", "🟡", "🔴", "⚪"
    reasoning: str

@dataclass
class StanceResult:
    """Result of stance detection analysis."""
    stance: str  # "support", "contradict", "neutral"
    confidence: float
    evidence_sentences: List[str]
    reasoning: str

@dataclass
class ClaimAnalysisResult:
    """Result of claim analysis."""
    original_claim: str
    extracted_phrases: List[str]
    news_articles: List[Dict[str, Any]]
    google_factcheck_result: Optional[Dict[str, Any]] = None
    confidence_badge: Optional[ConfidenceBadge] = None
    analysis_summary: str = ""
    sources_checked: List[str] = None
    processing_time: float = 0.0
    claim_type: str = "general"
    stance_results: List[StanceResult] = None
    semantic_scores: List[float] = None

    def __post_init__(self):
        if self.sources_checked is None:
            self.sources_checked = []
        if self.stance_results is None:
            self.stance_results = []
        if self.semantic_scores is None:
            self.semantic_scores = []

class EnhancedClaimAnalyzer:
    """Enhanced claim analyzer using semantic search and improved stance detection."""
    
    def __init__(self, news_api_key: str, google_api_key: str):
        """
        Initialize the enhanced claim analyzer.
        
        Args:
            news_api_key: News API key
            google_api_key: Google Fact Check API key
        """
        self.news_handler = None
        self.google_factcheck = None
        self.verifier = None
        self.semantic_model = None
        
        # Rate limiting configuration
        self.request_delay = 0.5  # 500ms between requests (2 requests/sec max)
        self.max_retries = 3
        self.retry_delay = 2.0  # 2 seconds on retry
        
        # Stance detection thresholds
        self.support_threshold = 0.6
        self.contradict_threshold = 0.6
        
        # Verdict aggregation thresholds
        self.support_percentage_threshold = 0.4  # 40% support = Likely True
        self.contradict_percentage_threshold = 0.4  # 40% contradict = Likely False
        
        # Initialize News API handler
        if COMPONENTS_AVAILABLE:
            try:
                self.news_handler = NewsHandler(news_api_key)
                logger.info("News API handler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize News API handler: {e}")
        
        # Initialize Google Fact Check API
        if COMPONENTS_AVAILABLE:
            try:
                self.google_factcheck = GoogleFactCheckAPI(google_api_key)
                logger.info("Google Fact Check API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Fact Check API: {e}")
        
        # Initialize Enhanced Verifier
        if COMPONENTS_AVAILABLE:
            try:
                self.verifier = EnhancedVerifier(google_api_key=google_api_key)
                logger.info("Enhanced Verifier initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced Verifier: {e}")
        
        # Initialize semantic search model
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic search model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize semantic model: {e}")
                self.semantic_model = None
    
    def classify_claim_type(self, claim: str) -> str:
        """Classify the type of claim for better search strategies."""
        claim_lower = claim.lower()
        
        # Causal claims
        causal_patterns = [
            r'\bcauses?\b', r'\bcaused\b', r'\bleads?\s+to\b', r'\bled\s+to\b',
            r'\bresults?\s+in\b', r'\btrigger(s|ed)?\b', r'\bcreates?\b',
            r'\bproduces?\b', r'\bgenerates?\b', r'\binduces?\b'
        ]
        
        # Factual claims
        factual_patterns = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bhas\b', r'\bhave\b',
            r'\bcontains?\b', r'\bconsists?\s+of\b', r'\bcomprises?\b', r'\bamounts?\s+to\b'
        ]
        
        # Prediction claims
        prediction_patterns = [
            r'\bwill\b', r'\bgoing\s+to\b', r'\bgonna\b', r'\bwould\b', r'\bcould\b',
            r'\bmight\b', r'\bmay\b', r'\bshall\b', r'\bexpected\b', r'\bforecast\b'
        ]
        
        # Opinion indicators
        opinion_patterns = [
            r'\bbelieve\b', r'\bthink\b', r'\bfeel\b', r'\bopinion\b', r'\bview\b',
            r'\bseem\b', r'\bappear\b', r'\blook\s+like\b', r'\bsound\s+like\b'
        ]
        
        # Count matches
        causal_count = sum(1 for pattern in causal_patterns if re.search(pattern, claim_lower))
        factual_count = sum(1 for pattern in factual_patterns if re.search(pattern, claim_lower))
        prediction_count = sum(1 for pattern in prediction_patterns if re.search(pattern, claim_lower))
        opinion_count = sum(1 for pattern in opinion_patterns if re.search(pattern, claim_lower))
        
        # Determine claim type
        if causal_count > 0:
            return "causal"
        elif prediction_count > 0:
            return "prediction"
        elif opinion_count > 0:
            return "opinion"
        elif factual_count > 0:
            return "factual"
        else:
            return "general"
    
    def extract_search_phrases(self, claim: str) -> List[str]:
        """
        Extract meaningful phrases from a claim for News API search.
        Uses advanced NLP techniques and News API best practices.
        """
        claim_type = self.classify_claim_type(claim)
        
        # Clean the claim
        claim_clean = re.sub(r'[^\w\s]', ' ', claim)
        words = claim_clean.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'very', 'really', 'quite', 'just', 'only', 'even', 'still', 'also', 'too', 'as',
            'so', 'than', 'more', 'most', 'less', 'least', 'much', 'many', 'few', 'several'
        }
        
        # Filter out stop words and short words
        meaningful_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        # Create phrases based on claim type
        phrases = []
        
        if claim_type == "causal":
            # For causal claims, focus on the cause and effect
            phrases = self._extract_causal_phrases(claim, meaningful_words)
        elif claim_type == "factual":
            # For factual claims, focus on the main entities and facts
            phrases = self._extract_factual_phrases(claim, meaningful_words)
        elif claim_type == "prediction":
            # For predictions, focus on the subject and predicted outcome
            phrases = self._extract_prediction_phrases(claim, meaningful_words)
        else:
            # General approach: create n-grams and key phrases
            phrases = self._extract_general_phrases(claim, meaningful_words)
        
        # Add the original claim as a phrase if it's not too long
        if len(claim.split()) <= 8:
            phrases.append(claim)
        
        # Remove duplicates and limit to top phrases
        unique_phrases = []
        seen = set()
        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            if phrase_lower not in seen and len(phrase_lower) > 3:
                seen.add(phrase_lower)
                unique_phrases.append(phrase)
        
        return unique_phrases[:5]  # Limit to top 5 phrases
    
    def _extract_causal_phrases(self, claim: str, words: List[str]) -> List[str]:
        """Extract phrases for causal claims."""
        phrases = []
        
        # Look for cause-effect patterns
        causal_patterns = [
            r'(\w+(?:\s+\w+){0,3})\s+(?:causes?|caused|leads?\s+to|led\s+to|results?\s+in)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:trigger(s|ed)?|creates?|produces?|generates?)\s+(\w+(?:\s+\w+){0,3})',
        ]
        
        for pattern in causal_patterns:
            matches = re.finditer(pattern, claim, re.IGNORECASE)
            for match in matches:
                cause = match.group(1).strip()
                effect = match.group(2).strip()
                phrases.extend([cause, effect, f"{cause} {effect}"])
        
        # Add key entities
        phrases.extend(self._extract_key_entities(claim, words))
        
        return phrases
    
    def _extract_factual_phrases(self, claim: str, words: List[str]) -> List[str]:
        """Extract phrases for factual claims."""
        phrases = []
        
        # Look for factual patterns
        factual_patterns = [
            r'(\w+(?:\s+\w+){0,3})\s+(?:is|are|was|were|has|have)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:contains?|consists?\s+of|comprises?)\s+(\w+(?:\s+\w+){0,3})',
        ]
        
        for pattern in factual_patterns:
            matches = re.finditer(pattern, claim, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                phrases.extend([subject, predicate, f"{subject} {predicate}"])
        
        # Add key entities
        phrases.extend(self._extract_key_entities(claim, words))
        
        return phrases
    
    def _extract_prediction_phrases(self, claim: str, words: List[str]) -> List[str]:
        """Extract phrases for prediction claims."""
        phrases = []
        
        # Look for prediction patterns
        prediction_patterns = [
            r'(\w+(?:\s+\w+){0,3})\s+(?:will|going\s+to|gonna|would|could)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:expected|forecast|predicted)\s+(\w+(?:\s+\w+){0,3})',
        ]
        
        for pattern in prediction_patterns:
            matches = re.finditer(pattern, claim, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                prediction = match.group(2).strip()
                phrases.extend([subject, prediction, f"{subject} {prediction}"])
        
        # Add key entities
        phrases.extend(self._extract_key_entities(claim, words))
        
        return phrases
    
    def _extract_general_phrases(self, claim: str, words: List[str]) -> List[str]:
        """Extract general phrases using n-grams and key entities."""
        phrases = []
        
        # Create n-grams (2-4 word combinations)
        for n in range(2, min(5, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                if len(phrase) > 3:
                    phrases.append(phrase)
        
        # Add key entities
        phrases.extend(self._extract_key_entities(claim, words))
        
        return phrases
    
    def _extract_key_entities(self, claim: str, words: List[str]) -> List[str]:
        """Extract key entities from the claim."""
        entities = []
        
        # Look for proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
        entities.extend(proper_nouns)
        
        # Look for numbers and measurements
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand))?\b', claim, re.IGNORECASE)
        entities.extend(numbers)
        
        # Look for specific terms (diseases, technologies, etc.)
        specific_terms = re.findall(r'\b(?:covid|coronavirus|vaccine|5g|ai|artificial\s+intelligence|climate\s+change|global\s+warming)\b', claim, re.IGNORECASE)
        entities.extend(specific_terms)
        
        return entities
    
    def search_news_with_semantic_ranking(self, claim: str, phrases: List[str], max_articles: int = 20) -> List[Dict[str, Any]]:
        """
        Search news articles using semantic ranking and deduplication.
        
        Args:
            claim: Original claim
            phrases: Search phrases
            max_articles: Maximum number of articles to retrieve
            
        Returns:
            List of news articles ranked by semantic similarity
        """
        if not self.news_handler:
            logger.warning("News handler not available")
            return []
        
        all_articles = []
        
        # Limit the number of phrases to avoid rate limiting
        max_phrases = min(3, len(phrases))  # Only use top 3 phrases to avoid rate limits
        selected_phrases = phrases[:max_phrases]
        
        # Search for each phrase with proper rate limiting
        for i, phrase in enumerate(selected_phrases):
            try:
                logger.info(f"Searching news for phrase {i+1}/{len(selected_phrases)}: {phrase}")
                
                # Use different search strategies based on phrase length
                if len(phrase.split()) <= 2:
                    # Short phrases: use exact search
                    articles = self._search_with_retry(
                        f'"{phrase}"', 
                        max_results=max_articles//len(selected_phrases),
                        days_back=30
                    )
                else:
                    # Longer phrases: use broader search
                    articles = self._search_with_retry(
                        phrase, 
                        max_results=max_articles//len(selected_phrases),
                        days_back=30
                    )
                
                for article in articles:
                    article_dict = {
                        "title": article.title,
                        "description": article.description,
                        "url": article.url,
                        "source": article.source,
                        "published_at": article.published_at,
                        "relevance_score": article.relevance_score,
                        "search_phrase": phrase,
                        "content_hash": self._get_content_hash(article.title + article.description)
                    }
                    all_articles.append(article_dict)
                
                # Rate limiting: wait between requests (except for the last one)
                if i < len(selected_phrases) - 1:
                    logger.info(f"Rate limiting: waiting {self.request_delay}s before next request...")
                    time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error searching news for phrase '{phrase}': {e}")
                continue
        
        # Deduplicate articles based on content hash
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Apply semantic ranking if available
        if self.semantic_model and unique_articles:
            unique_articles = self._rank_by_semantic_similarity(claim, unique_articles)
        
        return unique_articles[:max_articles]
    
    def _get_content_hash(self, content: str) -> str:
        """Generate a hash for content deduplication."""
        return hashlib.md5(content.lower().encode()).hexdigest()
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on content hash."""
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            content_hash = article.get('content_hash', '')
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_articles.append(article)
        
        logger.info(f"Deduplicated {len(articles)} articles to {len(unique_articles)} unique articles")
        return unique_articles
    
    def _rank_by_semantic_similarity(self, claim: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank articles by semantic similarity to the claim."""
        try:
            # Prepare texts for embedding
            texts = [claim] + [f"{article['title']} {article['description']}" for article in articles]
            
            # Get embeddings
            embeddings = self.semantic_model.encode(texts)
            
            # Calculate similarities
            claim_embedding = embeddings[0].reshape(1, -1)
            article_embeddings = embeddings[1:]
            
            similarities = cosine_similarity(claim_embedding, article_embeddings)[0]
            
            # Add semantic scores to articles
            for i, article in enumerate(articles):
                article['semantic_score'] = float(similarities[i])
            
            # Sort by semantic score
            ranked_articles = sorted(articles, key=lambda x: x['semantic_score'], reverse=True)
            
            logger.info(f"Ranked {len(articles)} articles by semantic similarity")
            return ranked_articles
            
        except Exception as e:
            logger.error(f"Error in semantic ranking: {e}")
            return articles
    
    def _search_with_retry(self, query: str, max_results: int, days_back: int) -> List[NewsArticle]:
        """
        Search with retry logic for rate limiting.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            days_back: Days back to search
            
        Returns:
            List of news articles
        """
        for attempt in range(self.max_retries):
            try:
                return self.news_handler.search_news(query, max_results=max_results, days_back=days_back)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}/{self.max_retries}, waiting {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    # Increase delay for next attempt
                    self.retry_delay = min(self.retry_delay * 2, 10.0)
                else:
                    logger.error(f"Error in search attempt {attempt + 1}: {e}")
                    break
        
        logger.error(f"Failed to search after {self.max_retries} attempts")
        return []
    
    def detect_stance_improved(self, claim: str, article: Dict[str, Any]) -> StanceResult:
        """
        Improved stance detection with better logic and rule-based signals.
        """
        title = article['title'].lower()
        description = article['description'].lower()
        text_to_analyze = f"{title} {description}"
        
        # Rule-based signals for explicit contradictions
        contradiction_keywords = [
            'false', 'debunked', 'misleading', 'inaccurate', 'wrong', 'fake', 'hoax',
            'myth', 'disproven', 'refuted', 'denied', 'rejected', 'contradicted',
            'fact check', 'misinformation', 'disinformation', 'not true', 'untrue',
            'no evidence', 'no link', 'no connection', 'does not cause', 'does not lead to'
        ]
        
        # Check for explicit contradictions
        for keyword in contradiction_keywords:
            if keyword in text_to_analyze:
                evidence_sentences = self._extract_evidence_sentences(description, keyword)
                return StanceResult(
                    stance="contradict",
                    confidence=0.9,
                    evidence_sentences=evidence_sentences,
                    reasoning=f"Article explicitly contains contradiction keyword: '{keyword}'"
                )
        
        # Support keywords for causal claims
        support_keywords = [
            'confirmed', 'verified', 'true', 'accurate', 'correct', 'proven', 'fact',
            'study shows', 'research confirms', 'evidence shows', 'scientists say',
            'official', 'announced', 'reported', 'confirmed by', 'verified by'
        ]
        
        # Check for explicit support
        for keyword in support_keywords:
            if keyword in text_to_analyze:
                evidence_sentences = self._extract_evidence_sentences(description, keyword)
                return StanceResult(
                    stance="support",
                    confidence=0.8,
                    evidence_sentences=evidence_sentences,
                    reasoning=f"Article explicitly contains support keyword: '{keyword}'"
                )
        
        # Semantic analysis for causal claims
        if self._is_causal_claim(claim):
            stance, confidence, evidence = self._analyze_causal_stance(claim, text_to_analyze)
            return StanceResult(
                stance=stance,
                confidence=confidence,
                evidence_sentences=evidence,
                reasoning=f"Semantic analysis of causal relationship"
            )
        
        # Default to neutral if no clear stance detected
        return StanceResult(
            stance="neutral",
            confidence=0.5,
            evidence_sentences=[],
            reasoning="No clear stance detected"
        )
    
    def _is_causal_claim(self, claim: str) -> bool:
        """Check if claim is causal."""
        causal_words = ['causes', 'caused', 'leads to', 'led to', 'results in', 'resulted in']
        return any(word in claim.lower() for word in causal_words)
    
    def _analyze_causal_stance(self, claim: str, article_text: str) -> Tuple[str, float, List[str]]:
        """
        Analyze stance for causal claims by looking for evidence of cause-effect relationships.
        """
        claim_lower = claim.lower()
        
        # Extract cause and effect from claim
        cause_effect = self._extract_cause_effect(claim_lower)
        if not cause_effect:
            return "neutral", 0.5, []
        
        cause, effect = cause_effect
        
        # Look for evidence of the effect in the article
        effect_evidence = self._find_effect_evidence(effect, article_text)
        
        if effect_evidence:
            # If we find evidence of the effect, it supports the causal claim
            return "support", 0.7, effect_evidence
        
        # Look for evidence that contradicts the causal relationship
        contradiction_evidence = self._find_contradiction_evidence(cause, effect, article_text)
        
        if contradiction_evidence:
            return "contradict", 0.7, contradiction_evidence
        
        return "neutral", 0.5, []
    
    def _extract_cause_effect(self, claim: str) -> Optional[Tuple[str, str]]:
        """Extract cause and effect from a causal claim."""
        patterns = [
            r'(\w+(?:\s+\w+){0,3})\s+(?:causes?|caused|leads?\s+to|led\s+to|results?\s+in)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+){0,3})\s+(?:trigger(s|ed)?|creates?|produces?|generates?)\s+(\w+(?:\s+\w+){0,3})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        
        return None
    
    def _find_effect_evidence(self, effect: str, article_text: str) -> List[str]:
        """Find evidence of the effect in the article text."""
        evidence_sentences = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', article_text)
        
        # Look for sentences that mention the effect or related concepts
        effect_words = effect.split()
        effect_keywords = self._get_related_keywords(effect)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains effect words or related keywords
            if any(word in sentence_lower for word in effect_words + effect_keywords):
                if len(sentence.strip()) > 10:  # Only include substantial sentences
                    evidence_sentences.append(sentence.strip())
        
        return evidence_sentences[:3]  # Limit to top 3 evidence sentences
    
    def _get_related_keywords(self, concept: str) -> List[str]:
        """Get related keywords for a concept."""
        concept_lower = concept.lower()
        
        # Define related keywords for common effects
        related_keywords = {
            'destruction': ['damage', 'devastation', 'loss', 'destruction', 'wreckage', 'ruin'],
            'deaths': ['death', 'casualty', 'fatality', 'killed', 'died', 'victim'],
            'floods': ['flooding', 'water', 'inundation', 'submerged', 'overflow'],
            'autism': ['autism', 'autistic', 'developmental disorder'],
            'jobs': ['employment', 'work', 'career', 'job loss', 'unemployment'],
            'weather': ['storm', 'rain', 'wind', 'temperature', 'climate'],
        }
        
        for key, keywords in related_keywords.items():
            if key in concept_lower:
                return keywords
        
        return []
    
    def _find_contradiction_evidence(self, cause: str, effect: str, article_text: str) -> List[str]:
        """Find evidence that contradicts the causal relationship."""
        evidence_sentences = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', article_text)
        
        # Look for sentences that explicitly deny the causal relationship
        contradiction_patterns = [
            r'no\s+link\s+between',
            r'no\s+connection\s+between',
            r'does\s+not\s+cause',
            r'not\s+caused\s+by',
            r'no\s+evidence\s+that',
            r'debunked',
            r'false',
            r'myth'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains contradiction patterns
            if any(re.search(pattern, sentence_lower) for pattern in contradiction_patterns):
                if len(sentence.strip()) > 10:
                    evidence_sentences.append(sentence.strip())
        
        return evidence_sentences[:3]
    
    def _extract_evidence_sentences(self, text: str, keyword: str) -> List[str]:
        """Extract sentences containing the keyword as evidence."""
        sentences = re.split(r'[.!?]+', text)
        evidence_sentences = []
        
        for sentence in sentences:
            if keyword in sentence.lower() and len(sentence.strip()) > 10:
                evidence_sentences.append(sentence.strip())
        
        return evidence_sentences[:2]  # Limit to top 2 evidence sentences
    
    def analyze_claim_with_google_factcheck(self, claim: str) -> Optional[Dict[str, Any]]:
        """Analyze claim using Google Fact Check API."""
        if not self.google_factcheck:
            logger.warning("Google Fact Check API not available")
            return None
        
        try:
            logger.info(f"Analyzing claim with Google Fact Check: {claim[:50]}...")
            result = self.google_factcheck.verify_claim(claim)
            
            if result:
                return {
                    "verdict": result.verdict,
                    "confidence": result.confidence,
                    "publisher": result.publisher,
                    "rating": result.rating,
                    "url": result.url,
                    "explanation": result.explanation
                }
            
        except Exception as e:
            logger.error(f"Error analyzing claim with Google Fact Check: {e}")
        
        return None
    
    def calculate_confidence_improved(self, claim: str, news_articles: List[Dict[str, Any]], 
                                    stance_results: List[StanceResult], 
                                    google_factcheck_result: Optional[Dict[str, Any]] = None) -> Tuple[float, str]:
        """
        Calculate confidence score with improved aggregation logic.
        """
        if not news_articles and not google_factcheck_result:
            return 0.0, "No evidence available for analysis"
        
        # Count stances
        stance_counts = defaultdict(int)
        for stance_result in stance_results:
            stance_counts[stance_result.stance] += 1
        
        total_articles = len(stance_results)
        
        # Calculate percentages
        support_percentage = stance_counts['support'] / total_articles if total_articles > 0 else 0
        contradict_percentage = stance_counts['contradict'] / total_articles if total_articles > 0 else 0
        neutral_percentage = stance_counts['neutral'] / total_articles if total_articles > 0 else 0
        
        # Check for scientific consensus claims
        if self._is_scientific_consensus_claim(claim):
            return self._handle_scientific_consensus_claim(claim, support_percentage, contradict_percentage, google_factcheck_result)
        
        # Apply improved verdict aggregation logic
        if google_factcheck_result:
            # Google Fact Check result overrides news analysis
            google_verdict = google_factcheck_result['verdict']
            google_confidence = google_factcheck_result['confidence']
            
            if google_verdict == "REFUTED":
                return 0.1, f"Google Fact Check: REFUTED ({google_confidence:.1%}). News analysis: {support_percentage:.1%} support, {contradict_percentage:.1%} contradict"
            elif google_verdict == "SUPPORTED":
                return 0.9, f"Google Fact Check: SUPPORTED ({google_confidence:.1%}). News analysis: {support_percentage:.1%} support, {contradict_percentage:.1%} contradict"
            else:
                # Mixed or unclear from Google Fact Check
                if support_percentage > self.support_percentage_threshold:
                    return 0.7, f"Google Fact Check: {google_verdict} ({google_confidence:.1%}). News analysis: {support_percentage:.1%} support"
                elif contradict_percentage > self.contradict_percentage_threshold:
                    return 0.3, f"Google Fact Check: {google_verdict} ({google_confidence:.1%}). News analysis: {contradict_percentage:.1%} contradict"
                else:
                    return 0.5, f"Google Fact Check: {google_verdict} ({google_confidence:.1%}). News analysis: {neutral_percentage:.1%} neutral"
        
        # No Google Fact Check result, use news analysis
        if support_percentage > self.support_percentage_threshold:
            confidence = 0.6 + (support_percentage * 0.3)  # Base 0.6 + up to 0.3 for high support
            return confidence, f"News analysis: {support_percentage:.1%} support, {contradict_percentage:.1%} contradict"
        elif contradict_percentage > self.contradict_percentage_threshold:
            confidence = 0.1 + (contradict_percentage * 0.2)  # Base 0.1 + up to 0.2 for high contradiction
            return confidence, f"News analysis: {contradict_percentage:.1%} contradict, {support_percentage:.1%} support"
        else:
            # Mixed or unclear evidence
            if total_articles == 0:
                return 0.0, "No news articles found to analyze"
            else:
                return 0.5, f"Mixed evidence: {support_percentage:.1%} support, {contradict_percentage:.1%} contradict, {neutral_percentage:.1%} neutral"
    
    def _is_scientific_consensus_claim(self, claim: str) -> bool:
        """Check if claim is about a topic with strong scientific consensus."""
        claim_lower = claim.lower()
        
        # Scientific consensus topics
        consensus_topics = [
            'vaccine autism', 'vaccines cause autism', 'vaccine autism link',
            'earth flat', 'earth is flat', 'flat earth',
            'climate change hoax', 'global warming hoax', 'climate change fake',
            '5g coronavirus', '5g covid', '5g causes coronavirus',
            'moon landing fake', 'moon landing hoax'
        ]
        
        return any(topic in claim_lower for topic in consensus_topics)
    
    def _handle_scientific_consensus_claim(self, claim: str, support_percentage: float, 
                                         contradict_percentage: float, 
                                         google_factcheck_result: Optional[Dict[str, Any]]) -> Tuple[float, str]:
        """Handle claims with strong scientific consensus."""
        claim_lower = claim.lower()
        
        # Default to refuted for consensus claims unless overwhelming evidence suggests otherwise
        if google_factcheck_result:
            google_verdict = google_factcheck_result['verdict']
            if google_verdict == "REFUTED":
                return 0.1, f"Scientific consensus + Google Fact Check: REFUTED"
            elif google_verdict == "SUPPORTED":
                return 0.9, f"Google Fact Check: SUPPORTED (unusual for consensus claim)"
        
        # For consensus claims, high support percentage might indicate misinformation
        if support_percentage > 0.3:
            return 0.1, f"Scientific consensus claim with {support_percentage:.1%} support (likely misinformation)"
        else:
            return 0.1, f"Scientific consensus claim: refuted by scientific evidence"
    
    def create_confidence_badge_improved(self, confidence: float, reasoning: str, 
                                       total_articles: int) -> ConfidenceBadge:
        """Create a confidence badge with improved logic."""
        if total_articles == 0:
            return ConfidenceBadge(
                level="Not Enough Info",
                confidence=0.0,
                color="gray",
                emoji="⚪",
                reasoning="No evidence available for analysis"
            )
        elif confidence >= 0.7:
            return ConfidenceBadge(
                level="Likely True",
                confidence=confidence,
                color="green",
                emoji="🟢",
                reasoning=reasoning
            )
        elif confidence >= 0.4:
            return ConfidenceBadge(
                level="Unclear",
                confidence=confidence,
                color="yellow",
                emoji="🟡",
                reasoning=reasoning
            )
        else:
            return ConfidenceBadge(
                level="Likely False",
                confidence=confidence,
                color="red",
                emoji="🔴",
                reasoning=reasoning
            )
    
    def analyze_claim(self, claim: str) -> ClaimAnalysisResult:
        """
        Analyze a claim using enhanced semantic search and improved stance detection.
        """
        import time
        start_time = time.time()
        
        logger.info(f"Analyzing claim: {claim}")
        
        # Step 1: Classify claim type
        claim_type = self.classify_claim_type(claim)
        logger.info(f"Claim type: {claim_type}")
        
        # Step 2: Extract search phrases
        phrases = self.extract_search_phrases(claim)
        logger.info(f"Extracted phrases: {phrases}")
        
        # Step 3: Search news articles with semantic ranking
        news_articles = self.search_news_with_semantic_ranking(claim, phrases)
        logger.info(f"Found {len(news_articles)} news articles")
        
        # Step 4: Check Google Fact Check API
        google_result = self.analyze_claim_with_google_factcheck(claim)
        
        # Step 5: Perform improved stance detection
        stance_results = []
        semantic_scores = []
        
        for article in news_articles:
            stance_result = self.detect_stance_improved(claim, article)
            stance_results.append(stance_result)
            semantic_scores.append(article.get('semantic_score', 0.0))
        
        # Step 6: Calculate confidence with improved logic
        confidence, reasoning = self.calculate_confidence_improved(
            claim, news_articles, stance_results, google_result
        )
        
        # Step 7: Create confidence badge
        confidence_badge = self.create_confidence_badge_improved(
            confidence, reasoning, len(news_articles)
        )
        
        # Step 8: Create analysis summary
        analysis_summary = f"""
Enhanced Claim Analysis Summary:
- Original Claim: {claim}
- Claim Type: {claim_type}
- Search Phrases: {', '.join(phrases)}
- News Articles Found: {len(news_articles)}
- Google Fact Check: {'Available' if google_result else 'Not available'}
- Confidence: {confidence_badge.emoji} {confidence_badge.level} ({confidence:.1%})
- Reasoning: {reasoning}
- Stance Distribution: {dict(stance_counts) if (stance_counts := {s: len([r for r in stance_results if r.stance == s]) for s in ['support', 'contradict', 'neutral']}) else 'N/A'}
"""
        
        processing_time = time.time() - start_time
        
        return ClaimAnalysisResult(
            original_claim=claim,
            extracted_phrases=phrases,
            news_articles=news_articles,
            google_factcheck_result=google_result,
            confidence_badge=confidence_badge,
            analysis_summary=analysis_summary,
            sources_checked=["News API", "Google Fact Check API"] if google_result else ["News API"],
            processing_time=processing_time,
            claim_type=claim_type,
            stance_results=stance_results,
            semantic_scores=semantic_scores
        )

def create_enhanced_claim_analyzer(news_api_key: str, google_api_key: str) -> EnhancedClaimAnalyzer:
    """Factory function to create an enhanced claim analyzer."""
    return EnhancedClaimAnalyzer(news_api_key, google_api_key)
