from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HighlightedSpan:
    """Represents a highlighted span of text with its relevance score and type."""
    start: int
    end: int
    text: str
    relevance_score: float
    highlight_type: str  # "support", "refute", "neutral"
    confidence: float


@dataclass
class CitationHighlight:
    """Represents a citation with highlighted spans."""
    citation_text: str
    highlighted_spans: List[HighlightedSpan]
    overall_support_score: float
    overall_refute_score: float
    stance: str  # "SUPPORTED", "REFUTED", "NOT ENOUGH INFO"


class CitationHighlighter:
    """Highlights relevant parts of evidence text that support or refute claims."""
    
    def __init__(self):
        # Keywords that indicate support
        self.support_keywords = [
            "confirm", "verified", "true", "accurate", "correct", "proven",
            "evidence shows", "study finds", "research indicates", "according to",
            "official", "confirmed", "validated", "authentic", "genuine"
        ]
        
        # Keywords that indicate refutation
        self.refute_keywords = [
            "false", "fake", "hoax", "debunked", "disproven", "incorrect",
            "misleading", "untrue", "fabricated", "denied", "rejected",
            "no evidence", "not true", "false claim", "misinformation"
        ]
        
        # Negation patterns
        self.negation_patterns = [
            r"\bno\b", r"\bnot\b", r"n't\b", r"\bnever\b", r"\bdeny\b",
            r"\brefute\b", r"\bdisprove\b", r"\bdebunk\b"
        ]
        
        # Compile regex patterns
        self.negation_regex = re.compile("|".join(self.negation_patterns), re.IGNORECASE)
    
    def highlight_citation(self, claim: str, citation_text: str, stance: str, 
                          stance_probabilities: Dict[str, float]) -> CitationHighlight:
        """Highlight relevant parts of a citation based on the claim and stance."""
        
        # Extract key terms from the claim
        claim_terms = self._extract_key_terms(claim)
        
        # Find relevant spans
        highlighted_spans = []
        
        # Find supporting evidence spans
        support_spans = self._find_support_spans(citation_text, claim_terms)
        for span in support_spans:
            highlighted_spans.append(HighlightedSpan(
                start=span[0],
                end=span[1],
                text=span[2],
                relevance_score=span[3],
                highlight_type="support",
                confidence=stance_probabilities.get("SUPPORTED", 0.0)
            ))
        
        # Find refuting evidence spans
        refute_spans = self._find_refute_spans(citation_text, claim_terms)
        for span in refute_spans:
            highlighted_spans.append(HighlightedSpan(
                start=span[0],
                end=span[1],
                text=span[2],
                relevance_score=span[3],
                highlight_type="refute",
                confidence=stance_probabilities.get("REFUTED", 0.0)
            ))
        
        # Find neutral/context spans
        neutral_spans = self._find_neutral_spans(citation_text, claim_terms)
        for span in neutral_spans:
            highlighted_spans.append(HighlightedSpan(
                start=span[0],
                end=span[1],
                text=span[2],
                relevance_score=span[3],
                highlight_type="neutral",
                confidence=stance_probabilities.get("NOT ENOUGH INFO", 0.0)
            ))
        
        # Sort spans by relevance score
        highlighted_spans.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Calculate overall scores
        overall_support = sum(s.relevance_score for s in highlighted_spans if s.highlight_type == "support")
        overall_refute = sum(s.relevance_score for s in highlighted_spans if s.highlight_type == "refute")
        
        return CitationHighlight(
            citation_text=citation_text,
            highlighted_spans=highlighted_spans,
            overall_support_score=overall_support,
            overall_refute_score=overall_refute,
            stance=stance
        )
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for matching."""
        # Remove common stop words and extract meaningful terms
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms
    
    def _find_support_spans(self, text: str, claim_terms: List[str]) -> List[Tuple[int, int, str, float]]:
        """Find spans that support the claim."""
        spans = []
        
        # Look for support keywords
        for keyword in self.support_keywords:
            for match in re.finditer(rf'\b{re.escape(keyword)}\b', text, re.IGNORECASE):
                # Check if this keyword appears near claim terms
                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 100)
                context = text[context_start:context_end]
                
                # Calculate relevance based on proximity to claim terms
                relevance = self._calculate_relevance(context, claim_terms)
                if relevance > 0.3:  # Threshold for relevance
                    spans.append((match.start(), match.end(), match.group(), relevance))
        
        # Look for claim term matches with positive context
        for term in claim_terms:
            for match in re.finditer(rf'\b{re.escape(term)}\b', text, re.IGNORECASE):
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                
                # Check if context is positive
                if self._is_positive_context(context):
                    relevance = self._calculate_relevance(context, claim_terms)
                    spans.append((match.start(), match.end(), match.group(), relevance))
        
        return spans
    
    def _find_refute_spans(self, text: str, claim_terms: List[str]) -> List[Tuple[int, int, str, float]]:
        """Find spans that refute the claim."""
        spans = []
        
        # Look for refute keywords
        for keyword in self.refute_keywords:
            for match in re.finditer(rf'\b{re.escape(keyword)}\b', text, re.IGNORECASE):
                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 100)
                context = text[context_start:context_end]
                
                relevance = self._calculate_relevance(context, claim_terms)
                if relevance > 0.3:
                    spans.append((match.start(), match.end(), match.group(), relevance))
        
        # Look for claim term matches with negative context
        for term in claim_terms:
            for match in re.finditer(rf'\b{re.escape(term)}\b', text, re.IGNORECASE):
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                
                # Check if context is negative
                if self._is_negative_context(context):
                    relevance = self._calculate_relevance(context, claim_terms)
                    spans.append((match.start(), match.end(), match.group(), relevance))
        
        return spans
    
    def _find_neutral_spans(self, text: str, claim_terms: List[str]) -> List[Tuple[int, int, str, float]]:
        """Find neutral/contextual spans."""
        spans = []
        
        # Look for claim terms in neutral context
        for term in claim_terms:
            for match in re.finditer(rf'\b{re.escape(term)}\b', text, re.IGNORECASE):
                context_start = max(0, match.start() - 30)
                context_end = min(len(text), match.end() + 30)
                context = text[context_start:context_end]
                
                # Check if context is neutral
                if not self._is_positive_context(context) and not self._is_negative_context(context):
                    relevance = self._calculate_relevance(context, claim_terms) * 0.5  # Lower relevance for neutral
                    if relevance > 0.2:
                        spans.append((match.start(), match.end(), match.group(), relevance))
        
        return spans
    
    def _is_positive_context(self, context: str) -> bool:
        """Check if context is positive/supporting."""
        positive_indicators = ["confirm", "true", "accurate", "verified", "proven", "evidence"]
        return any(indicator in context.lower() for indicator in positive_indicators)
    
    def _is_negative_context(self, context: str) -> bool:
        """Check if context is negative/refuting."""
        negative_indicators = ["false", "fake", "hoax", "debunked", "disproven", "incorrect"]
        return any(indicator in context.lower() for indicator in negative_indicators)
    
    def _calculate_relevance(self, context: str, claim_terms: List[str]) -> float:
        """Calculate relevance score based on term overlap."""
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        overlap = sum(1 for term in claim_terms if term in context_words)
        return min(1.0, overlap / max(1, len(claim_terms)))
    
    def format_highlighted_text(self, highlight: CitationHighlight) -> str:
        """Format highlighted text with HTML-like tags for display."""
        text = highlight.citation_text
        spans = sorted(highlight.highlighted_spans, key=lambda x: x.start)
        
        # Build formatted text with highlights
        result = ""
        last_end = 0
        
        for span in spans:
            # Add text before this span
            result += text[last_end:span.start]
            
            # Add highlighted span
            if span.highlight_type == "support":
                result += f"<span class='highlight-support' data-score='{span.relevance_score:.2f}'>{span.text}</span>"
            elif span.highlight_type == "refute":
                result += f"<span class='highlight-refute' data-score='{span.relevance_score:.2f}'>{span.text}</span>"
            else:  # neutral
                result += f"<span class='highlight-neutral' data-score='{span.relevance_score:.2f}'>{span.text}</span>"
            
            last_end = span.end
        
        # Add remaining text
        result += text[last_end:]
        
        return result


def highlight_citations(claim: str, citations: List[Dict[str, any]]) -> List[CitationHighlight]:
    """Highlight multiple citations for a claim."""
    highlighter = CitationHighlighter()
    highlighted_citations = []
    
    for citation in citations:
        citation_text = citation.get("text", citation.get("snippet", ""))
        stance = citation.get("stance", "NOT ENOUGH INFO")
        stance_probs = citation.get("stance_probabilities", {})
        
        if citation_text:
            highlight = highlighter.highlight_citation(claim, citation_text, stance, stance_probs)
            highlighted_citations.append(highlight)
    
    return highlighted_citations
