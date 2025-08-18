"""
Output builder for Phase 5 manipulation detection results.

Formats manipulation detection results into UX-friendly JSON structures
for frontend consumption, including:
- Manipulation cues chips
- Educational content
- User interface elements
"""

import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from .rules import ManipulationResult, ManipulationType
from .llm_detector import LLMDetectionResult
from .prebunk_generator import PrebunkTip, EducationalCard


@dataclass
class ManipulationCue:
    """A manipulation cue for UI display."""
    type: str
    label: str
    description: str
    severity: str  # "low", "medium", "high"
    confidence: float
    color: str  # CSS color code
    icon: str
    tooltip: str


@dataclass
class ManipulationSummary:
    """Summary of manipulation detection results."""
    total_detections: int
    manipulation_types: Dict[str, int]
    overall_confidence: float
    primary_concerns: List[str]
    risk_level: str  # "low", "medium", "high"
    recommendations: List[str]


@dataclass
class UXOutput:
    """Complete UX output for frontend consumption."""
    manipulation_cues: List[ManipulationCue]
    summary: ManipulationSummary
    educational_content: Dict[str, Any]
    prebunk_tips: List[Dict[str, Any]]
    interactive_elements: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ManipulationOutputBuilder:
    """Builder for manipulation detection UX outputs."""
    
    def __init__(self):
        self.severity_colors = {
            "low": "#28a745",      # Green
            "medium": "#ffc107",   # Yellow
            "high": "#dc3545"      # Red
        }
        
        self.manipulation_icons = {
            "clickbait": "📰",
            "fear_appeal": "😨",
            "cherry_picking": "🍒",
            "false_authority": "👨‍⚕️",
            "missing_context": "❓",
            "emotional_manipulation": "💔",
            "urgency": "⏰",
            "conspiracy": "🔍"
        }
    
    def build_manipulation_cues(self, rule_results: List[ManipulationResult], 
                               llm_results: Optional[LLMDetectionResult] = None) -> List[ManipulationCue]:
        """
        Build manipulation cues from detection results.
        
        Args:
            rule_results: Results from rule-based detection
            llm_results: Optional results from LLM detection
            
        Returns:
            List of manipulation cues for UI display
        """
        cues = []
        
        # Process rule-based results
        for result in rule_results:
            cue = self._create_cue_from_result(result)
            cues.append(cue)
        
        # Process LLM results if available
        if llm_results and llm_results.manipulation_types:
            for manipulation_type in llm_results.manipulation_types:
                # Avoid duplicates
                if not any(cue.type == manipulation_type for cue in cues):
                    cue = self._create_cue_from_llm_result(manipulation_type, llm_results.confidence)
                    cues.append(cue)
        
        return cues
    
    def _create_cue_from_result(self, result: ManipulationResult) -> ManipulationCue:
        """Create a manipulation cue from a rule-based result."""
        
        severity = self._determine_severity(result.confidence)
        color = self.severity_colors[severity]
        icon = self.manipulation_icons.get(result.manipulation_type.value, "⚠️")
        
        return ManipulationCue(
            type=result.manipulation_type.value,
            label=self._get_manipulation_label(result.manipulation_type),
            description=result.description,
            severity=severity,
            confidence=result.confidence,
            color=color,
            icon=icon,
            tooltip=f"{result.description} (Confidence: {result.confidence:.1%})"
        )
    
    def _create_cue_from_llm_result(self, manipulation_type: str, confidence: float) -> ManipulationCue:
        """Create a manipulation cue from an LLM result."""
        
        severity = self._determine_severity(confidence)
        color = self.severity_colors[severity]
        icon = self.manipulation_icons.get(manipulation_type, "⚠️")
        
        return ManipulationCue(
            type=manipulation_type,
            label=self._get_manipulation_label_from_string(manipulation_type),
            description=f"AI-detected {manipulation_type.replace('_', ' ')}",
            severity=severity,
            confidence=confidence,
            color=color,
            icon=icon,
            tooltip=f"AI-detected {manipulation_type.replace('_', ' ')} (Confidence: {confidence:.1%})"
        )
    
    def _determine_severity(self, confidence: float) -> str:
        """Determine severity level based on confidence score."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_manipulation_label(self, manipulation_type: ManipulationType) -> str:
        """Get user-friendly label for manipulation type."""
        labels = {
            ManipulationType.CLICKBAIT: "Clickbait",
            ManipulationType.FEAR_APPEAL: "Fear Appeal",
            ManipulationType.CHERRY_PICKING: "Cherry Picking",
            ManipulationType.FALSE_AUTHORITY: "False Authority",
            ManipulationType.MISSING_CONTEXT: "Missing Context",
            ManipulationType.EMOTIONAL_MANIPULATION: "Emotional Manipulation",
            ManipulationType.URGENCY: "False Urgency",
            ManipulationType.CONSPIRACY: "Conspiracy Language"
        }
        return labels.get(manipulation_type, manipulation_type.value.replace('_', ' ').title())
    
    def _get_manipulation_label_from_string(self, manipulation_type: str) -> str:
        """Get user-friendly label for manipulation type string."""
        return manipulation_type.replace('_', ' ').title()
    
    def build_summary(self, rule_results: List[ManipulationResult], 
                     llm_results: Optional[LLMDetectionResult] = None) -> ManipulationSummary:
        """
        Build manipulation summary from detection results.
        
        Args:
            rule_results: Results from rule-based detection
            llm_results: Optional results from LLM detection
            
        Returns:
            ManipulationSummary with overall analysis
        """
        total_detections = len(rule_results)
        manipulation_types = {}
        total_confidence = 0.0
        
        # Process rule-based results
        for result in rule_results:
            manipulation_type = result.manipulation_type.value
            if manipulation_type not in manipulation_types:
                manipulation_types[manipulation_type] = 0
            manipulation_types[manipulation_type] += 1
            total_confidence += result.confidence
        
        # Process LLM results if available
        if llm_results:
            total_detections += len(llm_results.manipulation_types)
            for manipulation_type in llm_results.manipulation_types:
                if manipulation_type not in manipulation_types:
                    manipulation_types[manipulation_type] = 0
                manipulation_types[manipulation_type] += 1
            total_confidence += llm_results.confidence
        
        # Calculate overall confidence
        overall_confidence = total_confidence / total_detections if total_detections > 0 else 0.0
        
        # Determine primary concerns
        primary_concerns = []
        for manipulation_type, count in manipulation_types.items():
            if count >= 2:  # Multiple detections of same type
                primary_concerns.append(manipulation_type.replace('_', ' ').title())
        
        # Determine risk level
        risk_level = self._determine_risk_level(total_detections, overall_confidence)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(manipulation_types, overall_confidence)
        
        return ManipulationSummary(
            total_detections=total_detections,
            manipulation_types=manipulation_types,
            overall_confidence=overall_confidence,
            primary_concerns=primary_concerns,
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    def _determine_risk_level(self, total_detections: int, overall_confidence: float) -> str:
        """Determine overall risk level."""
        if total_detections >= 5 or overall_confidence >= 0.8:
            return "high"
        elif total_detections >= 3 or overall_confidence >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, manipulation_types: Dict[str, int], 
                                 overall_confidence: float) -> List[str]:
        """Generate recommendations based on detected manipulation types."""
        recommendations = []
        
        if "clickbait" in manipulation_types:
            recommendations.append("Look for factual headlines instead of sensational language")
        
        if "fear_appeal" in manipulation_types:
            recommendations.append("Seek evidence-based information, not fear-based claims")
        
        if "cherry_picking" in manipulation_types:
            recommendations.append("Look for the full context and all available evidence")
        
        if "false_authority" in manipulation_types:
            recommendations.append("Verify expert credentials and check if other experts agree")
        
        if "missing_context" in manipulation_types:
            recommendations.append("Look for the complete story, not just selected parts")
        
        if "conspiracy" in manipulation_types:
            recommendations.append("Look for evidence, not just conspiracy claims")
        
        # Add general recommendations
        if overall_confidence >= 0.7:
            recommendations.append("This content shows multiple manipulation techniques - verify with multiple sources")
        
        if not recommendations:
            recommendations.append("Check multiple credible sources to verify information")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def build_educational_content(self, prebunk_tips: List[PrebunkTip]) -> Dict[str, Any]:
        """
        Build educational content from prebunking tips.
        
        Args:
            prebunk_tips: List of prebunking tips
            
        Returns:
            Dictionary with educational content for UI
        """
        educational_content = {
            "tips": [],
            "interactive_elements": [],
            "learning_path": []
        }
        
        for tip in prebunk_tips:
            tip_content = {
                "id": f"tip_{tip.tip_type.value}",
                "title": tip.title,
                "description": tip.description,
                "examples": tip.examples,
                "red_flags": tip.red_flags,
                "how_to_spot": tip.how_to_spot,
                "what_to_do": tip.what_to_do,
                "difficulty": tip.difficulty,
                "estimated_time": tip.estimated_read_time,
                "type": tip.tip_type.value
            }
            educational_content["tips"].append(tip_content)
            
            # Add to learning path
            educational_content["learning_path"].append({
                "step": len(educational_content["learning_path"]) + 1,
                "title": tip.title,
                "type": tip.tip_type.value,
                "difficulty": tip.difficulty,
                "time": tip.estimated_read_time
            })
        
        # Add interactive elements
        educational_content["interactive_elements"] = [
            {
                "type": "quiz",
                "title": "Test Your Knowledge",
                "description": "Take a quick quiz to test your understanding",
                "estimated_time": 60
            },
            {
                "type": "practice",
                "title": "Practice Scenarios",
                "description": "Practice identifying manipulation techniques",
                "estimated_time": 120
            },
            {
                "type": "examples",
                "title": "Real Examples",
                "description": "See real-world examples of manipulation",
                "estimated_time": 90
            }
        ]
        
        return educational_content
    
    def build_complete_output(self, rule_results: List[ManipulationResult],
                             llm_results: Optional[LLMDetectionResult] = None,
                             prebunk_tips: Optional[List[PrebunkTip]] = None) -> UXOutput:
        """
        Build complete UX output from all detection results.
        
        Args:
            rule_results: Results from rule-based detection
            llm_results: Optional results from LLM detection
            prebunk_tips: Optional list of prebunking tips
            
        Returns:
            Complete UXOutput for frontend consumption
        """
        # Build manipulation cues
        manipulation_cues = self.build_manipulation_cues(rule_results, llm_results)
        
        # Build summary
        summary = self.build_summary(rule_results, llm_results)
        
        # Build educational content
        educational_content = {}
        if prebunk_tips:
            educational_content = self.build_educational_content(prebunk_tips)
        
        # Build prebunk tips for UI
        prebunk_tips_ui = []
        if prebunk_tips:
            for tip in prebunk_tips:
                tip_ui = {
                    "id": f"tip_{tip.tip_type.value}",
                    "title": tip.title,
                    "description": tip.description,
                    "examples": tip.examples,
                    "red_flags": tip.red_flags,
                    "how_to_spot": tip.how_to_spot,
                    "what_to_do": tip.what_to_do,
                    "difficulty": tip.difficulty,
                    "estimated_time": tip.estimated_read_time,
                    "type": tip.tip_type.value
                }
                prebunk_tips_ui.append(tip_ui)
        
        # Build interactive elements
        interactive_elements = [
            {
                "type": "manipulation_cues",
                "title": "Manipulation Cues",
                "description": "Visual indicators of detected manipulation",
                "data": [asdict(cue) for cue in manipulation_cues]
            },
            {
                "type": "educational_tips",
                "title": "Learn More",
                "description": "Educational content about manipulation techniques",
                "data": prebunk_tips_ui
            },
            {
                "type": "practice_quiz",
                "title": "Practice Quiz",
                "description": "Test your knowledge of manipulation techniques",
                "data": []
            }
        ]
        
        # Build metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "detection_methods": ["rule_based"],
            "total_analysis_time": 0.0
        }
        
        if llm_results:
            metadata["detection_methods"].append("llm_based")
        
        return UXOutput(
            manipulation_cues=manipulation_cues,
            summary=summary,
            educational_content=educational_content,
            prebunk_tips=prebunk_tips_ui,
            interactive_elements=interactive_elements,
            metadata=metadata
        )
    
    def to_json(self, output: UXOutput) -> str:
        """
        Convert UX output to JSON string.
        
        Args:
            output: UXOutput to convert
            
        Returns:
            JSON string representation
        """
        # Convert dataclasses to dictionaries
        output_dict = {
            "manipulation_cues": [asdict(cue) for cue in output.manipulation_cues],
            "summary": asdict(output.summary),
            "educational_content": output.educational_content,
            "prebunk_tips": output.prebunk_tips,
            "interactive_elements": output.interactive_elements,
            "metadata": output.metadata
        }
        
        return json.dumps(output_dict, indent=2, ensure_ascii=False)
    
    def to_dict(self, output: UXOutput) -> Dict[str, Any]:
        """
        Convert UX output to dictionary.
        
        Args:
            output: UXOutput to convert
            
        Returns:
            Dictionary representation
        """
        return {
            "manipulation_cues": [asdict(cue) for cue in output.manipulation_cues],
            "summary": asdict(output.summary),
            "educational_content": output.educational_content,
            "prebunk_tips": output.prebunk_tips,
            "interactive_elements": output.interactive_elements,
            "metadata": output.metadata
        }
