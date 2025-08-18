"""
Tests for output builder module.
"""

import pytest
import json
from src.manipulation.output_builder import (
    ManipulationOutputBuilder,
    ManipulationCue,
    ManipulationSummary,
    UXOutput
)
from src.manipulation.rules import ManipulationResult, ManipulationType
from src.manipulation.llm_detector import LLMDetectionResult
from src.manipulation.prebunk_generator import PrebunkTip, TipType


class TestManipulationOutputBuilder:
    """Test cases for ManipulationOutputBuilder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = ManipulationOutputBuilder()
    
    def test_initialization(self):
        """Test builder initialization."""
        assert self.builder is not None
        assert len(self.builder.severity_colors) == 3
        assert len(self.builder.manipulation_icons) == 8
    
    def test_create_cue_from_result(self):
        """Test creating manipulation cue from rule result."""
        result = ManipulationResult(
            manipulation_type=ManipulationType.CLICKBAIT,
            confidence=0.8,
            matched_text="You won't believe",
            description="Uses sensational language",
            suggestions=["Look for factual headlines"]
        )
        
        cue = self.builder._create_cue_from_result(result)
        
        assert isinstance(cue, ManipulationCue)
        assert cue.type == "clickbait"
        assert cue.label == "Clickbait"
        assert cue.description == "Uses sensational language"
        assert cue.confidence == 0.8
        assert cue.icon == "📰"
        assert "sensational language" in cue.tooltip
    
    def test_create_cue_from_llm_result(self):
        """Test creating manipulation cue from LLM result."""
        cue = self.builder._create_cue_from_llm_result("fear_appeal", 0.7)
        
        assert isinstance(cue, ManipulationCue)
        assert cue.type == "fear_appeal"
        assert cue.label == "Fear Appeal"
        assert cue.description == "AI-detected fear appeal"
        assert cue.confidence == 0.7
        assert cue.icon == "😨"
    
    def test_determine_severity(self):
        """Test severity determination based on confidence."""
        assert self.builder._determine_severity(0.9) == "high"
        assert self.builder._determine_severity(0.7) == "medium"
        assert self.builder._determine_severity(0.5) == "low"
        assert self.builder._determine_severity(0.3) == "low"
    
    def test_get_manipulation_label(self):
        """Test getting user-friendly labels for manipulation types."""
        assert self.builder._get_manipulation_label(ManipulationType.CLICKBAIT) == "Clickbait"
        assert self.builder._get_manipulation_label(ManipulationType.FEAR_APPEAL) == "Fear Appeal"
        assert self.builder._get_manipulation_label(ManipulationType.CHERRY_PICKING) == "Cherry Picking"
    
    def test_get_manipulation_label_from_string(self):
        """Test getting labels from string manipulation types."""
        assert self.builder._get_manipulation_label_from_string("clickbait") == "Clickbait"
        assert self.builder._get_manipulation_label_from_string("fear_appeal") == "Fear Appeal"
        assert self.builder._get_manipulation_label_from_string("unknown_type") == "Unknown Type"
    
    def test_build_manipulation_cues(self):
        """Test building manipulation cues from detection results."""
        rule_results = [
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                matched_text="You won't believe",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            )
        ]
        
        cues = self.builder.build_manipulation_cues(rule_results)
        
        assert len(cues) == 1
        assert cues[0].type == "clickbait"
        assert cues[0].confidence == 0.8
    
    def test_build_manipulation_cues_with_llm(self):
        """Test building manipulation cues with LLM results."""
        rule_results = [
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                matched_text="You won't believe",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            )
        ]
        
        llm_results = LLMDetectionResult(
            manipulation_types=["fear_appeal"],
            confidence=0.7,
            reasoning="Uses fear-based language",
            specific_examples=["dangerous"],
            suggestions=["Look for evidence"],
            bias_indicators=[],
            logical_fallacies=[]
        )
        
        cues = self.builder.build_manipulation_cues(rule_results, llm_results)
        
        assert len(cues) == 2
        cue_types = {cue.type for cue in cues}
        assert "clickbait" in cue_types
        assert "fear_appeal" in cue_types
    
    def test_build_summary(self):
        """Test building manipulation summary."""
        rule_results = [
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                matched_text="You won't believe",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            ),
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.9,
                matched_text="Shocking revelation",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            )
        ]
        
        summary = self.builder.build_summary(rule_results)
        
        assert isinstance(summary, ManipulationSummary)
        assert summary.total_detections == 2
        assert summary.manipulation_types["clickbait"] == 2
        assert abs(summary.overall_confidence - 0.85) < 0.001  # Handle floating point precision
        assert "Clickbait" in summary.primary_concerns
        assert len(summary.recommendations) > 0
    
    def test_determine_risk_level(self):
        """Test risk level determination."""
        assert self.builder._determine_risk_level(5, 0.8) == "high"
        assert self.builder._determine_risk_level(3, 0.6) == "medium"
        assert self.builder._determine_risk_level(1, 0.3) == "low"
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        manipulation_types = {"clickbait": 2, "fear_appeal": 1}
        recommendations = self.builder._generate_recommendations(manipulation_types, 0.7)
        
        assert len(recommendations) > 0
        assert any("factual headlines" in rec for rec in recommendations)
        assert any("evidence-based" in rec for rec in recommendations)
    
    def test_build_educational_content(self):
        """Test building educational content."""
        tips = [
            PrebunkTip(
                tip_type=TipType.CLICKBAIT,
                title="Don't Fall for Clickbait!",
                description="Uses sensational language",
                examples=["Example 1"],
                red_flags=["Flag 1"],
                how_to_spot=["Method 1"],
                what_to_do=["Action 1"],
                difficulty="easy",
                estimated_read_time=30
            )
        ]
        
        content = self.builder.build_educational_content(tips)
        
        assert "tips" in content
        assert "interactive_elements" in content
        assert "learning_path" in content
        assert len(content["tips"]) == 1
        assert len(content["interactive_elements"]) == 3
        assert len(content["learning_path"]) == 1
    
    def test_build_complete_output(self):
        """Test building complete UX output."""
        rule_results = [
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                matched_text="You won't believe",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            )
        ]
        
        tips = [
            PrebunkTip(
                tip_type=TipType.CLICKBAIT,
                title="Don't Fall for Clickbait!",
                description="Uses sensational language",
                examples=["Example 1"],
                red_flags=["Flag 1"],
                how_to_spot=["Method 1"],
                what_to_do=["Action 1"],
                difficulty="easy",
                estimated_read_time=30
            )
        ]
        
        output = self.builder.build_complete_output(rule_results, prebunk_tips=tips)
        
        assert isinstance(output, UXOutput)
        assert len(output.manipulation_cues) == 1
        assert output.summary.total_detections == 1
        assert len(output.prebunk_tips) == 1
        assert len(output.interactive_elements) == 3
        assert "timestamp" in output.metadata
    
    def test_to_json(self):
        """Test converting output to JSON string."""
        rule_results = [
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                matched_text="You won't believe",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            )
        ]
        
        output = self.builder.build_complete_output(rule_results)
        json_str = self.builder.to_json(output)
        
        assert isinstance(json_str, str)
        # Should be valid JSON
        json_data = json.loads(json_str)
        assert "manipulation_cues" in json_data
        assert "summary" in json_data
        assert "educational_content" in json_data
    
    def test_to_dict(self):
        """Test converting output to dictionary."""
        rule_results = [
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                matched_text="You won't believe",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            )
        ]
        
        output = self.builder.build_complete_output(rule_results)
        output_dict = self.builder.to_dict(output)
        
        assert isinstance(output_dict, dict)
        assert "manipulation_cues" in output_dict
        assert "summary" in output_dict
        assert "educational_content" in output_dict


class TestManipulationCue:
    """Test cases for ManipulationCue dataclass."""
    
    def test_cue_creation(self):
        """Test creating a ManipulationCue."""
        cue = ManipulationCue(
            type="clickbait",
            label="Clickbait",
            description="Uses sensational language",
            severity="high",
            confidence=0.8,
            color="#dc3545",
            icon="📰",
            tooltip="Uses sensational language (Confidence: 80.0%)"
        )
        
        assert cue.type == "clickbait"
        assert cue.label == "Clickbait"
        assert cue.description == "Uses sensational language"
        assert cue.severity == "high"
        assert cue.confidence == 0.8
        assert cue.color == "#dc3545"
        assert cue.icon == "📰"
        assert "80.0%" in cue.tooltip


class TestManipulationSummary:
    """Test cases for ManipulationSummary dataclass."""
    
    def test_summary_creation(self):
        """Test creating a ManipulationSummary."""
        summary = ManipulationSummary(
            total_detections=3,
            manipulation_types={"clickbait": 2, "fear_appeal": 1},
            overall_confidence=0.75,
            primary_concerns=["Clickbait"],
            risk_level="medium",
            recommendations=["Look for factual headlines"]
        )
        
        assert summary.total_detections == 3
        assert summary.manipulation_types["clickbait"] == 2
        assert summary.manipulation_types["fear_appeal"] == 1
        assert summary.overall_confidence == 0.75
        assert summary.primary_concerns == ["Clickbait"]
        assert summary.risk_level == "medium"
        assert summary.recommendations == ["Look for factual headlines"]


class TestUXOutput:
    """Test cases for UXOutput dataclass."""
    
    def test_output_creation(self):
        """Test creating a UXOutput."""
        cues = [
            ManipulationCue(
                type="clickbait",
                label="Clickbait",
                description="Uses sensational language",
                severity="high",
                confidence=0.8,
                color="#dc3545",
                icon="📰",
                tooltip="Uses sensational language (Confidence: 80.0%)"
            )
        ]
        
        summary = ManipulationSummary(
            total_detections=1,
            manipulation_types={"clickbait": 1},
            overall_confidence=0.8,
            primary_concerns=[],
            risk_level="high",
            recommendations=[]
        )
        
        output = UXOutput(
            manipulation_cues=cues,
            summary=summary,
            educational_content={},
            prebunk_tips=[],
            interactive_elements=[],
            metadata={"timestamp": "2024-01-01T00:00:00"}
        )
        
        assert len(output.manipulation_cues) == 1
        assert output.summary.total_detections == 1
        assert output.summary.overall_confidence == 0.8
        assert output.summary.risk_level == "high"
        assert "timestamp" in output.metadata


class TestColorAndIconMapping:
    """Test cases for color and icon mappings."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = ManipulationOutputBuilder()
    
    def test_severity_colors(self):
        """Test severity color mapping."""
        assert self.builder.severity_colors["low"] == "#28a745"  # Green
        assert self.builder.severity_colors["medium"] == "#ffc107"  # Yellow
        assert self.builder.severity_colors["high"] == "#dc3545"  # Red
    
    def test_manipulation_icons(self):
        """Test manipulation icon mapping."""
        assert self.builder.manipulation_icons["clickbait"] == "📰"
        assert self.builder.manipulation_icons["fear_appeal"] == "😨"
        assert self.builder.manipulation_icons["cherry_picking"] == "🍒"
        assert self.builder.manipulation_icons["false_authority"] == "👨‍⚕️"
        assert self.builder.manipulation_icons["missing_context"] == "❓"
        assert self.builder.manipulation_icons["emotional_manipulation"] == "💔"
        assert self.builder.manipulation_icons["urgency"] == "⏰"
        assert self.builder.manipulation_icons["conspiracy"] == "🔍"
    
    def test_unknown_manipulation_type(self):
        """Test handling of unknown manipulation types."""
        result = ManipulationResult(
            manipulation_type=ManipulationType.CLICKBAIT,  # This should work
            confidence=0.8,
            matched_text="test",
            description="test",
            suggestions=[]
        )
        
        cue = self.builder._create_cue_from_result(result)
        assert cue.icon == "📰"  # Should get the correct icon
        
        # Test with unknown string type
        cue = self.builder._create_cue_from_llm_result("unknown_type", 0.5)
        assert cue.icon == "⚠️"  # Should get default icon


if __name__ == "__main__":
    pytest.main([__file__])
