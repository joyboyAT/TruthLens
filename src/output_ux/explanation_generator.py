from __future__ import annotations

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ExplanationType(Enum):
    """Types of explanations for misleading content."""
    MANIPULATION_TECHNIQUE = "manipulation_technique"
    MISSING_CONTEXT = "missing_context"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    LOW_CONFIDENCE = "low_confidence"
    BIASED_SOURCE = "biased_source"
    OUTDATED_INFO = "outdated_info"


@dataclass
class Explanation:
    """Represents an explanation for why content might be misleading."""
    explanation_type: ExplanationType
    title: str
    description: str
    severity: str  # "low", "medium", "high"
    suggestions: List[str]
    examples: List[str]


class ExplanationGenerator:
    """Generates explanations for why content might be misleading."""
    
    def __init__(self):
        self.explanations = self._initialize_explanations()
    
    def _initialize_explanations(self) -> Dict[str, Explanation]:
        """Initialize explanation templates."""
        return {
            "clickbait": Explanation(
                explanation_type=ExplanationType.MANIPULATION_TECHNIQUE,
                title="Sensational Language Detected",
                description="This content uses exaggerated or sensational language designed to grab attention rather than inform accurately.",
                severity="medium",
                suggestions=[
                    "Look for more neutral, factual reporting on this topic",
                    "Check if the claims are supported by credible sources",
                    "Be skeptical of overly dramatic language"
                ],
                examples=[
                    "You won't believe what happened next!",
                    "Shocking revelation that will change everything!"
                ]
            ),
            
            "fear_appeal": Explanation(
                explanation_type=ExplanationType.MANIPULATION_TECHNIQUE,
                title="Fear-Based Manipulation",
                description="This content uses fear to manipulate emotions and bypass rational thinking.",
                severity="high",
                suggestions=[
                    "Seek out balanced, factual information about the topic",
                    "Look for scientific consensus rather than alarmist claims",
                    "Check if the risks are being exaggerated"
                ],
                examples=[
                    "This dangerous chemical is in your food!",
                    "Deadly virus spreading rapidly - act now!"
                ]
            ),
            
            "cherry_picking": Explanation(
                explanation_type=ExplanationType.MANIPULATION_TECHNIQUE,
                title="Selective Evidence Presentation",
                description="This content presents only evidence that supports a particular viewpoint while ignoring contradictory information.",
                severity="high",
                suggestions=[
                    "Look for comprehensive reviews of the evidence",
                    "Check if there are conflicting studies or data",
                    "Seek out balanced perspectives from multiple sources"
                ],
                examples=[
                    "This study proves vaccines are dangerous",
                    "This data shows climate change is fake"
                ]
            ),
            
            "false_authority": Explanation(
                explanation_type=ExplanationType.MANIPULATION_TECHNIQUE,
                title="Unverified Authority Claims",
                description="This content appeals to authority figures without proper verification of their expertise or claims.",
                severity="medium",
                suggestions=[
                    "Verify the credentials and expertise of cited authorities",
                    "Check if the authority's views represent scientific consensus",
                    "Look for peer-reviewed research rather than individual opinions"
                ],
                examples=[
                    "Doctor says vaccines cause autism",
                    "Scientist reveals shocking truth"
                ]
            ),
            
            "missing_context": Explanation(
                explanation_type=ExplanationType.MISSING_CONTEXT,
                title="Incomplete Information",
                description="This content lacks important context that could change the interpretation of the information.",
                severity="medium",
                suggestions=[
                    "Look for more comprehensive coverage of the topic",
                    "Check the original source for full context",
                    "Seek out background information on the issue"
                ],
                examples=[
                    "Out-of-context quotes or statistics",
                    "Partial information that creates misleading impressions"
                ]
            ),
            
            "conflicting_evidence": Explanation(
                explanation_type=ExplanationType.CONFLICTING_EVIDENCE,
                title="Mixed Evidence",
                description="Available evidence on this topic is conflicting or inconclusive.",
                severity="low",
                suggestions=[
                    "Look for systematic reviews or meta-analyses",
                    "Check the quality and methodology of studies",
                    "Consider that the topic may still be under investigation"
                ],
                examples=[
                    "Some studies show X, others show Y",
                    "Research is ongoing and results are mixed"
                ]
            ),
            
            "low_confidence": Explanation(
                explanation_type=ExplanationType.LOW_CONFIDENCE,
                title="Low Confidence Assessment",
                description="Our analysis shows low confidence in the accuracy of this claim due to limited or poor quality evidence.",
                severity="medium",
                suggestions=[
                    "Look for more recent and comprehensive research",
                    "Check if there are better quality studies available",
                    "Be cautious about drawing strong conclusions"
                ],
                examples=[
                    "Limited evidence available",
                    "Poor quality or outdated studies"
                ]
            ),
            
            "biased_source": Explanation(
                explanation_type=ExplanationType.BIASED_SOURCE,
                title="Potentially Biased Source",
                description="The source of this information may have a bias or agenda that could affect the accuracy of the content.",
                severity="medium",
                suggestions=[
                    "Look for information from neutral, fact-checking organizations",
                    "Check multiple sources with different perspectives",
                    "Verify claims with primary sources when possible"
                ],
                examples=[
                    "Partisan news sources",
                    "Organizations with clear agendas"
                ]
            )
        }
    
    def generate_explanation(self, 
                           manipulation_cues: List[str],
                           stance: str,
                           confidence: float,
                           evidence_quality: str = "medium",
                           source_bias: Optional[str] = None) -> List[Explanation]:
        """Generate explanations for why content might be misleading."""
        explanations = []
        
        # Add manipulation technique explanations
        for cue in manipulation_cues:
            cue_key = cue.lower().replace(" ", "_").replace("-", "_")
            if cue_key in self.explanations:
                explanations.append(self.explanations[cue_key])
        
        # Add confidence-based explanation
        if confidence < 0.4:
            explanations.append(self.explanations["low_confidence"])
        
        # Add stance-based explanations
        if stance == "REFUTED" and confidence > 0.7:
            # If strongly refuted, add missing context explanation
            explanations.append(self.explanations["missing_context"])
        
        # Add source bias explanation if provided
        if source_bias and source_bias.lower() in ["high", "medium"]:
            explanations.append(self.explanations["biased_source"])
        
        # Add conflicting evidence if confidence is moderate
        if 0.4 <= confidence <= 0.6:
            explanations.append(self.explanations["conflicting_evidence"])
        
        return explanations
    
    def format_explanation_for_ux(self, explanations: List[Explanation]) -> Dict[str, Any]:
        """Format explanations for user interface display."""
        if not explanations:
            return {
                "has_explanations": False,
                "explanations": []
            }
        
        formatted_explanations = []
        for exp in explanations:
            formatted_explanations.append({
                "type": exp.explanation_type.value,
                "title": exp.title,
                "description": exp.description,
                "severity": exp.severity,
                "suggestions": exp.suggestions,
                "examples": exp.examples
            })
        
        return {
            "has_explanations": True,
            "explanations": formatted_explanations,
            "primary_concern": formatted_explanations[0] if formatted_explanations else None
        }


class UserExplanationLayer:
    """Main class for Phase 5 - User Explanation Layer integration."""
    
    def __init__(self):
        self.explanation_generator = ExplanationGenerator()
    
    def generate_complete_explanation(self,
                                    claim: str,
                                    stance: str,
                                    confidence: float,
                                    manipulation_cues: List[str],
                                    evidence_list: List[Dict[str, Any]],
                                    source_quality: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete user explanation including all Phase 5 components."""
        
        # Generate "Why misleading" explanations
        explanations = self.explanation_generator.generate_explanation(
            manipulation_cues=manipulation_cues,
            stance=stance,
            confidence=confidence,
            evidence_quality=source_quality.get("quality", "medium"),
            source_bias=source_quality.get("bias")
        )
        
        # Format explanations for UX
        explanation_data = self.explanation_generator.format_explanation_for_ux(explanations)
        
        # Generate summary explanation
        summary = self._generate_summary_explanation(claim, stance, confidence, manipulation_cues)
        
        return {
            "claim": claim,
            "stance": stance,
            "confidence": confidence,
            "summary_explanation": summary,
            "detailed_explanations": explanation_data,
            "manipulation_cues": manipulation_cues,
            "evidence_count": len(evidence_list),
            "source_quality": source_quality
        }
    
    def _generate_summary_explanation(self,
                                    claim: str,
                                    stance: str,
                                    confidence: float,
                                    manipulation_cues: List[str]) -> str:
        """Generate a concise summary explanation."""
        
        if stance == "SUPPORTED" and confidence > 0.7:
            return f"This claim appears to be supported by credible evidence with high confidence ({confidence:.1%})."
        
        elif stance == "REFUTED" and confidence > 0.7:
            return f"This claim appears to be false based on credible evidence with high confidence ({confidence:.1%})."
        
        elif stance == "NOT ENOUGH INFO" or confidence < 0.4:
            return f"There is insufficient evidence to determine the accuracy of this claim (confidence: {confidence:.1%})."
        
        else:
            base_explanation = f"This claim has mixed evidence with moderate confidence ({confidence:.1%})."
            
            if manipulation_cues:
                cue_names = [cue.replace("_", " ").title() for cue in manipulation_cues[:2]]
                base_explanation += f" The content shows signs of {', '.join(cue_names)}."
            
            return base_explanation
    
    def create_user_friendly_output(self, explanation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create user-friendly output for the interface."""
        
        # Determine overall assessment
        stance = explanation_data["stance"]
        confidence = explanation_data["confidence"]
        
        if stance == "SUPPORTED" and confidence > 0.7:
            assessment = "Likely True"
            color = "green"
        elif stance == "REFUTED" and confidence > 0.7:
            assessment = "Likely False"
            color = "red"
        else:
            assessment = "Unclear"
            color = "yellow"
        
        return {
            "assessment": assessment,
            "color": color,
            "confidence_percentage": f"{confidence:.1%}",
            "summary": explanation_data["summary_explanation"],
            "why_misleading": explanation_data["detailed_explanations"],
            "manipulation_techniques": explanation_data["manipulation_cues"],
            "evidence_summary": f"Analyzed {explanation_data['evidence_count']} sources",
            "recommendations": self._generate_recommendations(explanation_data)
        }
    
    def _generate_recommendations(self, explanation_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations for users."""
        recommendations = []
        
        # Base recommendations
        recommendations.append("Always verify claims with multiple credible sources")
        recommendations.append("Check the date of information to ensure it's current")
        
        # Add specific recommendations based on explanations
        if explanation_data["detailed_explanations"]["has_explanations"]:
            for exp in explanation_data["detailed_explanations"]["explanations"][:2]:  # Top 2
                recommendations.extend(exp["suggestions"][:2])  # Top 2 suggestions
        
        return list(set(recommendations))  # Remove duplicates


def generate_user_explanation(claim: str,
                            stance: str,
                            confidence: float,
                            manipulation_cues: List[str],
                            evidence_list: List[Dict[str, Any]],
                            source_quality: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to generate complete user explanation."""
    layer = UserExplanationLayer()
    explanation_data = layer.generate_complete_explanation(
        claim, stance, confidence, manipulation_cues, evidence_list, source_quality
    )
    return layer.create_user_friendly_output(explanation_data)
