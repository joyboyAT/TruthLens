#!/usr/bin/env python3
"""
Enhanced TruthLens Pipeline
Comprehensive integration of all improvements for better fact-checking accuracy.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Import enhanced components
try:
    from src.verification.enhanced_stance_classifier import EnhancedStanceClassifier, EnhancedStanceResult
    from src.verification.enhanced_verdict_aggregator import EnhancedVerdictAggregator, VerdictResult
    from src.verification.enhanced_factcheck_api import EnhancedFactCheckAPI, EnhancedFactCheckResult
    from src.evidence_retrieval.enhanced_semantic_search import EnhancedSemanticSearch, EnhancedSearchResult
    from src.news.news_handler import NewsHandler
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some enhanced components not available: {e}")
    COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EnhancedAnalysisResult:
    """Result of enhanced TruthLens analysis."""
    claim: str
    verdict: str
    confidence: float
    reasoning: str
    stance_distribution: Dict[str, int]
    stance_percentages: Dict[str, float]
    fact_check_result: Optional[Dict[str, Any]]
    news_articles: List[Dict[str, Any]]
    stance_results: List[Dict[str, Any]]
    search_summary: Dict[str, Any]
    processing_time: float
    analysis_timestamp: str
    evidence_summary: str
    rule_based_overrides: List[str]

class EnhancedTruthLensPipeline:
    """
    Enhanced TruthLens pipeline with comprehensive improvements.
    
    Key improvements implemented:
    1. Threshold tuning: support_prob > 0.6, contradict_prob > 0.6
    2. Enhanced stance detection with rule-based signals
    3. Improved verdict aggregation with 40% thresholds
    4. Multiple fact-checking sources integration
    5. Semantic search with deduplication and clustering
    6. Better causal reasoning for destruction/impact claims
    7. Scientific consensus handling
    """
    
    def __init__(self, news_api_key: str, google_api_key: str = None):
        """
        Initialize the enhanced TruthLens pipeline.
        
        Args:
            news_api_key: News API key
            google_api_key: Google Fact Check API key (optional)
        """
        self.news_api_key = news_api_key
        self.google_api_key = google_api_key
        
        # Initialize components
        self.news_handler = None
        self.stance_classifier = None
        self.verdict_aggregator = None
        self.fact_check_api = None
        self.semantic_search = None
        
        if COMPONENTS_AVAILABLE:
            self._initialize_components()
        else:
            logger.error("Enhanced components not available. Pipeline will not function properly.")
    
    def _initialize_components(self):
        """Initialize all enhanced components."""
        try:
            # Initialize News API handler
            self.news_handler = NewsHandler(self.news_api_key)
            logger.info("News API handler initialized")
            
            # Initialize enhanced stance classifier
            self.stance_classifier = EnhancedStanceClassifier()
            logger.info("Enhanced stance classifier initialized")
            
            # Initialize enhanced verdict aggregator
            self.verdict_aggregator = EnhancedVerdictAggregator()
            logger.info("Enhanced verdict aggregator initialized")
            
            # Initialize enhanced fact-check API
            if self.google_api_key:
                self.fact_check_api = EnhancedFactCheckAPI(self.google_api_key)
                logger.info("Enhanced fact-check API initialized")
            
            # Initialize enhanced semantic search
            self.semantic_search = EnhancedSemanticSearch()
            logger.info("Enhanced semantic search initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def analyze_claim(self, claim: str, max_articles: int = 20) -> EnhancedAnalysisResult:
        """
        Analyze a claim using the enhanced TruthLens pipeline.
        
        Args:
            claim: The claim to analyze
            max_articles: Maximum number of articles to analyze
            
        Returns:
            EnhancedAnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        logger.info(f"Starting enhanced analysis of claim: {claim}")
        
        try:
            # Step 1: Search for news articles
            news_articles = self._search_news_articles(claim, max_articles)
            logger.info(f"Found {len(news_articles)} news articles")
            
            # Step 2: Apply enhanced semantic search and ranking
            if self.semantic_search and news_articles:
                search_results = self.semantic_search.search_and_rank_articles(claim, news_articles, max_articles)
                ranked_articles = [result.article for result in search_results]
                search_summary = self.semantic_search.get_search_summary(search_results)
                logger.info("Applied enhanced semantic search and ranking")
            else:
                ranked_articles = news_articles
                search_summary = {"message": "Semantic search not available"}
            
            # Step 3: Check fact-checking sources
            fact_check_result = self._check_fact_check_sources(claim)
            
            # Step 4: Perform enhanced stance detection
            stance_results = self._detect_stances(claim, ranked_articles)
            logger.info(f"Completed stance detection for {len(stance_results)} articles")
            
            # Step 5: Aggregate verdict using enhanced logic
            verdict_result = self._aggregate_verdict(claim, stance_results, fact_check_result, len(ranked_articles))
            
            # Step 6: Generate evidence summary
            evidence_summary = self._generate_evidence_summary(stance_results, fact_check_result)
            
            # Step 7: Extract rule-based overrides
            rule_based_overrides = self._extract_rule_based_overrides(stance_results)
            
            processing_time = time.time() - start_time
            
            result = EnhancedAnalysisResult(
                claim=claim,
                verdict=verdict_result.verdict,
                confidence=verdict_result.confidence,
                reasoning=verdict_result.reasoning,
                stance_distribution=verdict_result.stance_distribution,
                stance_percentages=verdict_result.stance_percentages,
                fact_check_result=fact_check_result.to_dict() if fact_check_result else None,
                news_articles=ranked_articles,
                stance_results=[self._stance_result_to_dict(sr) for sr in stance_results],
                search_summary=search_summary,
                processing_time=processing_time,
                analysis_timestamp=datetime.now().isoformat(),
                evidence_summary=evidence_summary,
                rule_based_overrides=rule_based_overrides
            )
            
            logger.info(f"Enhanced analysis completed in {processing_time:.2f}s")
            logger.info(f"Final verdict: {result.verdict} (confidence: {result.confidence:.1%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced analysis: {e}")
            # Return error result
            return EnhancedAnalysisResult(
                claim=claim,
                verdict="Error",
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                stance_distribution={},
                stance_percentages={},
                fact_check_result=None,
                news_articles=[],
                stance_results=[],
                search_summary={"error": str(e)},
                processing_time=time.time() - start_time,
                analysis_timestamp=datetime.now().isoformat(),
                evidence_summary="Analysis failed due to error",
                rule_based_overrides=[]
            )
    
    def _search_news_articles(self, claim: str, max_articles: int) -> List[Dict[str, Any]]:
        """Search for news articles using News API."""
        if not self.news_handler:
            logger.warning("News handler not available")
            return []
        
        try:
            # Extract search phrases from claim
            search_phrases = self._extract_search_phrases(claim)
            
            all_articles = []
            for phrase in search_phrases[:3]:  # Use top 3 phrases
                try:
                    articles = self.news_handler.search_news(phrase, max_results=max_articles//3, days_back=30)
                    all_articles.extend(articles)
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error searching for phrase '{phrase}': {e}")
                    continue
            
            return all_articles
            
        except Exception as e:
            logger.error(f"Error in news search: {e}")
            return []
    
    def _extract_search_phrases(self, claim: str) -> List[str]:
        """Extract meaningful search phrases from a claim."""
        # Simple phrase extraction - in production you'd use more sophisticated NLP
        words = claim.split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]
        
        # Create phrases of 2-4 words
        phrases = []
        for i in range(len(meaningful_words)):
            for j in range(i+2, min(i+5, len(meaningful_words)+1)):
                phrase = ' '.join(meaningful_words[i:j])
                if len(phrase) > 5:  # Minimum phrase length
                    phrases.append(phrase)
        
        # Return top phrases by length (longer phrases are usually more specific)
        return sorted(phrases, key=len, reverse=True)[:5]
    
    def _check_fact_check_sources(self, claim: str) -> Optional[EnhancedFactCheckResult]:
        """Check multiple fact-checking sources."""
        if not self.fact_check_api:
            logger.info("Fact-check API not available")
            return None
        
        try:
            result = self.fact_check_api.get_best_fact_check(claim)
            if result:
                logger.info(f"Found fact-check result: {result.verdict} from {result.best_source['name']}")
            return result
        except Exception as e:
            logger.error(f"Error checking fact-check sources: {e}")
            return None
    
    def _detect_stances(self, claim: str, articles: List[Dict[str, Any]]) -> List[EnhancedStanceResult]:
        """Perform enhanced stance detection on articles."""
        if not self.stance_classifier:
            logger.warning("Stance classifier not available")
            return []
        
        stance_results = []
        for article in articles:
            try:
                stance_result = self.stance_classifier.classify_stance(claim, article)
                stance_results.append(stance_result)
            except Exception as e:
                logger.warning(f"Error in stance detection for article: {e}")
                # Create neutral stance as fallback
                stance_results.append(EnhancedStanceResult(
                    stance="neutral",
                    confidence=0.5,
                    evidence_sentences=[],
                    reasoning="Stance detection failed, defaulting to neutral"
                ))
        
        return stance_results
    
    def _aggregate_verdict(self, 
                          claim: str, 
                          stance_results: List[EnhancedStanceResult],
                          fact_check_result: Optional[EnhancedFactCheckResult],
                          total_articles: int) -> VerdictResult:
        """Aggregate verdict using enhanced logic."""
        if not self.verdict_aggregator:
            logger.warning("Verdict aggregator not available")
            # Create fallback verdict
            return VerdictResult(
                verdict="Error",
                confidence=0.0,
                reasoning="Verdict aggregation not available",
                stance_distribution={},
                stance_percentages={},
                evidence_summary="Component not available"
            )
        
        # Convert stance results to dict format
        stance_dicts = []
        for sr in stance_results:
            stance_dicts.append({
                'stance': sr.stance,
                'confidence': sr.confidence,
                'evidence_sentences': sr.evidence_sentences,
                'reasoning': sr.reasoning,
                'rule_based_override': sr.rule_based_override
            })
        
        # Convert fact-check result to dict if available
        fact_check_dict = None
        if fact_check_result:
            fact_check_dict = {
                'verdict': fact_check_result.verdict,
                'confidence': fact_check_result.confidence,
                'explanation': fact_check_result.explanation,
                'source': fact_check_result.best_source['name']
            }
        
        return self.verdict_aggregator.aggregate_verdict(
            claim, stance_dicts, fact_check_dict, total_articles
        )
    
    def _generate_evidence_summary(self, 
                                 stance_results: List[EnhancedStanceResult],
                                 fact_check_result: Optional[EnhancedFactCheckResult]) -> str:
        """Generate a comprehensive evidence summary."""
        if not self.verdict_aggregator:
            return "Evidence summary not available"
        
        stance_dicts = [self._stance_result_to_dict(sr) for sr in stance_results]
        return self.verdict_aggregator.get_evidence_summary(stance_dicts)
    
    def _extract_rule_based_overrides(self, stance_results: List[EnhancedStanceResult]) -> List[str]:
        """Extract rule-based overrides from stance results."""
        overrides = []
        for sr in stance_results:
            if sr.rule_based_override:
                overrides.append(sr.rule_based_override)
        return list(set(overrides))  # Remove duplicates
    
    def _stance_result_to_dict(self, stance_result: EnhancedStanceResult) -> Dict[str, Any]:
        """Convert stance result to dictionary format."""
        return {
            'stance': stance_result.stance,
            'confidence': stance_result.confidence,
            'evidence_sentences': stance_result.evidence_sentences,
            'reasoning': stance_result.reasoning,
            'rule_based_override': stance_result.rule_based_override,
            'model_probabilities': stance_result.model_probabilities
        }
    
    def get_analysis_summary(self, result: EnhancedAnalysisResult) -> str:
        """Generate a human-readable analysis summary."""
        summary_parts = [
            f"Enhanced TruthLens Analysis Summary",
            f"Claim: {result.claim}",
            f"Verdict: {result.verdict}",
            f"Confidence: {result.confidence:.1%}",
            f"Reasoning: {result.reasoning}",
            f"",
            f"Evidence Analysis:",
            f"- News Articles: {len(result.news_articles)}",
            f"- Stance Distribution: {dict(result.stance_distribution)}",
            f"- Stance Percentages: {dict(result.stance_percentages)}",
            f"",
            f"Fact-Check Integration:",
            f"- Fact-Check Result: {'Available' if result.fact_check_result else 'Not available'}",
            f"- Rule-Based Overrides: {', '.join(result.rule_based_overrides) if result.rule_based_overrides else 'None'}",
            f"",
            f"Search Quality:",
            f"- Search Summary: {result.search_summary}",
            f"",
            f"Processing:",
            f"- Time: {result.processing_time:.2f}s",
            f"- Timestamp: {result.analysis_timestamp}"
        ]
        
        return "\n".join(summary_parts)
    
    def export_results(self, result: EnhancedAnalysisResult, format: str = "json") -> str:
        """Export analysis results in specified format."""
        if format.lower() == "json":
            import json
            return json.dumps(result.__dict__, indent=2, default=str)
        elif format.lower() == "summary":
            return self.get_analysis_summary(result)
        else:
            raise ValueError(f"Unsupported export format: {format}")

def create_enhanced_pipeline(news_api_key: str, google_api_key: str = None) -> EnhancedTruthLensPipeline:
    """Factory function to create an enhanced TruthLens pipeline."""
    return EnhancedTruthLensPipeline(news_api_key, google_api_key)
