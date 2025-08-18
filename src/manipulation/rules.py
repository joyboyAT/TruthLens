"""
Rule-based manipulation detection for Phase 5.

Implements pattern matching for common manipulation techniques:
- Clickbait headlines
- Fear appeal
- Cherry-picking
- False authority
- Missing context
- Emotional manipulation
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ManipulationType(Enum):
    """Types of manipulation techniques."""
    CLICKBAIT = "clickbait"
    FEAR_APPEAL = "fear_appeal"
    CHERRY_PICKING = "cherry_picking"
    FALSE_AUTHORITY = "false_authority"
    MISSING_CONTEXT = "missing_context"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    URGENCY = "urgency"
    CONSPIRACY = "conspiracy"


@dataclass
class ManipulationRule:
    """A rule for detecting manipulation patterns."""
    name: str
    pattern: str
    manipulation_type: ManipulationType
    confidence: float
    description: str
    examples: List[str]


@dataclass
class ManipulationResult:
    """Result of manipulation detection."""
    manipulation_type: ManipulationType
    confidence: float
    matched_text: str
    description: str
    suggestions: List[str]


class ManipulationRuleDetector:
    """Rule-based detector for manipulation patterns."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[ManipulationRule]:
        """Initialize manipulation detection rules."""
        return [
            # Clickbait patterns
            ManipulationRule(
                name="Clickbait Headlines",
                pattern=r'\b(you won\'t believe|shocking|amazing|incredible|unbelievable|mind-blowing|jaw-dropping|stunning|outrageous|scandalous)\b',
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                description="Uses sensational language to grab attention",
                examples=["You won't believe what happened next!", "Shocking revelation about..."]
            ),
            
            # Fear appeal patterns
            ManipulationRule(
                name="Fear Appeal",
                pattern=r'\b(dangerous|deadly|killer|toxic|poison|threat|warning|alert|emergency|crisis|panic|scary|terrifying|horrifying)\b',
                manipulation_type=ManipulationType.FEAR_APPEAL,
                confidence=0.7,
                description="Uses fear to manipulate emotions",
                examples=["This dangerous chemical is in your food!", "Deadly virus spreading rapidly"]
            ),
            
            # Cherry-picking patterns
            ManipulationRule(
                name="Cherry Picking",
                pattern=r'\b(proves|shows|demonstrates|confirms|reveals)\b.*\b(and nothing else matters|and that\'s it|end of story|period|full stop)\b',
                manipulation_type=ManipulationType.CHERRY_PICKING,
                confidence=0.6,
                description="Selectively presents evidence to support a claim",
                examples=["This study proves vaccines are dangerous and nothing else matters", "This data shows climate change is fake"]
            ),
            
            # False authority patterns
            ManipulationRule(
                name="False Authority",
                pattern=r'\b(doctor|scientist|expert|researcher|professor|specialist|authority|official)\b.*\b(says|claims|reveals|admits|confesses|warns)\b',
                manipulation_type=ManipulationType.FALSE_AUTHORITY,
                confidence=0.7,
                description="Appeals to authority without proper verification",
                examples=["Doctor says vaccines cause autism", "Scientist reveals shocking truth"]
            ),
            
            # Missing context patterns
            ManipulationRule(
                name="Missing Context",
                pattern=r'\b(out of context|misleading|distorted|manipulated|edited|cropped|altered|modified)\b',
                manipulation_type=ManipulationType.MISSING_CONTEXT,
                confidence=0.8,
                description="Content lacks necessary context for proper understanding",
                examples=["This video was taken out of context", "Image was cropped to mislead"]
            ),
            
            # Emotional manipulation patterns
            ManipulationRule(
                name="Emotional Manipulation",
                pattern=r'\b(heartbreaking|devastating|tragic|sad|upsetting|disturbing|concerning|worrisome|alarming|shocking)\b',
                manipulation_type=ManipulationType.EMOTIONAL_MANIPULATION,
                confidence=0.6,
                description="Uses emotional language to manipulate feelings",
                examples=["Heartbreaking story of vaccine injury", "Tragic consequences of government policy"]
            ),
            
            # Urgency patterns
            ManipulationRule(
                name="Urgency",
                pattern=r'\b(urgent|immediate|now|quickly|hurry|limited time|last chance|don\'t wait|act now|time is running out)\b',
                manipulation_type=ManipulationType.URGENCY,
                confidence=0.7,
                description="Creates false urgency to pressure action",
                examples=["Act now before it's too late!", "Limited time offer - don't wait!"]
            ),
            
            # Conspiracy patterns
            ManipulationRule(
                name="Conspiracy Language",
                pattern=r'\b(conspiracy|cover-up|hidden|secret|suppressed|censored|mainstream media|establishment|elite|global)\b',
                manipulation_type=ManipulationType.CONSPIRACY,
                confidence=0.8,
                description="Uses conspiracy theory language",
                examples=["Mainstream media is hiding the truth", "Global elite conspiracy revealed"]
            )
        ]
    
    def detect_manipulation(self, text: str) -> List[ManipulationResult]:
        """
        Detect manipulation patterns in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected manipulation patterns
        """
        results = []
        text_lower = text.lower()
        
        for rule in self.rules:
            matches = re.finditer(rule.pattern, text_lower, re.IGNORECASE)
            
            for match in matches:
                matched_text = text[match.start():match.end()]
                
                # Generate suggestions based on manipulation type
                suggestions = self._generate_suggestions(rule.manipulation_type)
                
                result = ManipulationResult(
                    manipulation_type=rule.manipulation_type,
                    confidence=rule.confidence,
                    matched_text=matched_text,
                    description=rule.description,
                    suggestions=suggestions
                )
                results.append(result)
        
        return results
    
    def _generate_suggestions(self, manipulation_type: ManipulationType) -> List[str]:
        """Generate suggestions for addressing manipulation techniques."""
        suggestions_map = {
            ManipulationType.CLICKBAIT: [
                "Look for factual headlines instead of sensational language",
                "Check if the content delivers on the headline's promise",
                "Seek out balanced reporting from multiple sources"
            ],
            ManipulationType.FEAR_APPEAL: [
                "Look for evidence-based information, not fear-based claims",
                "Check if the threat is real or exaggerated",
                "Seek out expert opinions from multiple sources"
            ],
            ManipulationType.CHERRY_PICKING: [
                "Look for the full context and all available evidence",
                "Check if other studies or data contradict the claim",
                "Seek out comprehensive analysis, not selective information"
            ],
            ManipulationType.FALSE_AUTHORITY: [
                "Verify the expert's credentials and expertise",
                "Check if other experts agree with the claim",
                "Look for peer-reviewed research, not just opinions"
            ],
            ManipulationType.MISSING_CONTEXT: [
                "Look for the full story, not just selected parts",
                "Check the original source and full context",
                "Seek out complete information, not edited snippets"
            ],
            ManipulationType.EMOTIONAL_MANIPULATION: [
                "Focus on facts, not emotional language",
                "Look for evidence-based arguments",
                "Check if emotions are being used to distract from facts"
            ],
            ManipulationType.URGENCY: [
                "Take time to verify claims before acting",
                "Check if the urgency is real or manufactured",
                "Look for evidence that supports the urgent claim"
            ],
            ManipulationType.CONSPIRACY: [
                "Look for evidence, not just conspiracy claims",
                "Check if multiple credible sources support the claim",
                "Seek out fact-checking from reputable organizations"
            ]
        }
        
        return suggestions_map.get(manipulation_type, [
            "Look for evidence-based information",
            "Check multiple credible sources",
            "Verify claims with fact-checking organizations"
        ])
    
    def get_manipulation_summary(self, text: str) -> Dict:
        """
        Get a summary of detected manipulation patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Summary dictionary with manipulation types and counts
        """
        results = self.detect_manipulation(text)
        
        summary = {
            'total_detections': len(results),
            'manipulation_types': {},
            'confidence_score': 0.0,
            'suggestions': []
        }
        
        if results:
            # Count manipulation types
            for result in results:
                manipulation_type = result.manipulation_type.value
                if manipulation_type not in summary['manipulation_types']:
                    summary['manipulation_types'][manipulation_type] = 0
                summary['manipulation_types'][manipulation_type] += 1
            
            # Calculate average confidence
            total_confidence = sum(r.confidence for r in results)
            summary['confidence_score'] = total_confidence / len(results)
            
            # Collect unique suggestions
            all_suggestions = []
            for result in results:
                all_suggestions.extend(result.suggestions)
            summary['suggestions'] = list(set(all_suggestions))
        
        return summary
