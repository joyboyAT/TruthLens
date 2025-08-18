"""
Tests for manipulation rules module.
"""

import pytest
from src.manipulation.rules import (
    ManipulationRuleDetector, 
    ManipulationType, 
    ManipulationResult
)


class TestManipulationRuleDetector:
    """Test cases for ManipulationRuleDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ManipulationRuleDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector is not None
        assert len(self.detector.rules) > 0
    
    def test_clickbait_detection(self):
        """Test clickbait detection."""
        text = "You won't believe what happened next! This is shocking!"
        results = self.detector.detect_manipulation(text)
        
        assert len(results) > 0
        clickbait_results = [r for r in results if r.manipulation_type == ManipulationType.CLICKBAIT]
        assert len(clickbait_results) > 0
    
    def test_fear_appeal_detection(self):
        """Test fear appeal detection."""
        text = "This dangerous chemical is in your food and could kill you!"
        results = self.detector.detect_manipulation(text)
        
        assert len(results) > 0
        fear_results = [r for r in results if r.manipulation_type == ManipulationType.FEAR_APPEAL]
        assert len(fear_results) > 0
    
    def test_cherry_picking_detection(self):
        """Test cherry picking detection."""
        text = "This study proves vaccines are dangerous and nothing else matters."
        results = self.detector.detect_manipulation(text)
        
        assert len(results) > 0
        cherry_results = [r for r in results if r.manipulation_type == ManipulationType.CHERRY_PICKING]
        assert len(cherry_results) > 0
    
    def test_false_authority_detection(self):
        """Test false authority detection."""
        text = "Doctor says vaccines cause autism and you should believe him."
        results = self.detector.detect_manipulation(text)
        
        assert len(results) > 0
        authority_results = [r for r in results if r.manipulation_type == ManipulationType.FALSE_AUTHORITY]
        assert len(authority_results) > 0
    
    def test_conspiracy_detection(self):
        """Test conspiracy language detection."""
        text = "The mainstream media is hiding the truth about this conspiracy."
        results = self.detector.detect_manipulation(text)
        
        assert len(results) > 0
        conspiracy_results = [r for r in results if r.manipulation_type == ManipulationType.CONSPIRACY]
        assert len(conspiracy_results) > 0
    
    def test_multiple_manipulation_types(self):
        """Test detection of multiple manipulation types."""
        text = "You won't believe this shocking conspiracy! Doctor reveals dangerous truth!"
        results = self.detector.detect_manipulation(text)
        
        assert len(results) >= 2
        manipulation_types = {r.manipulation_type for r in results}
        assert len(manipulation_types) >= 2
    
    def test_no_manipulation_detected(self):
        """Test text with no manipulation."""
        text = "This is a factual statement about the weather. It is sunny today."
        results = self.detector.detect_manipulation(text)
        
        # Should have no or very few detections
        assert len(results) <= 1
    
    def test_manipulation_summary(self):
        """Test manipulation summary generation."""
        text = "You won't believe this shocking conspiracy! Doctor reveals dangerous truth!"
        summary = self.detector.get_manipulation_summary(text)
        
        assert 'total_detections' in summary
        assert 'manipulation_types' in summary
        assert 'confidence_score' in summary
        assert 'suggestions' in summary
        assert summary['total_detections'] > 0
    
    def test_suggestions_generation(self):
        """Test that suggestions are generated for each manipulation type."""
        text = "You won't believe this shocking conspiracy!"
        results = self.detector.detect_manipulation(text)
        
        for result in results:
            assert len(result.suggestions) > 0
            assert all(isinstance(s, str) for s in result.suggestions)


class TestManipulationTypes:
    """Test cases for ManipulationType enum."""
    
    def test_enum_values(self):
        """Test that all manipulation types are defined."""
        expected_types = {
            'CLICKBAIT',
            'FEAR_APPEAL', 
            'CHERRY_PICKING',
            'FALSE_AUTHORITY',
            'MISSING_CONTEXT',
            'EMOTIONAL_MANIPULATION',
            'URGENCY',
            'CONSPIRACY'
        }
        
        actual_types = {t.name for t in ManipulationType}
        assert actual_types == expected_types
    
    def test_enum_string_values(self):
        """Test that enum values are properly formatted strings."""
        for manipulation_type in ManipulationType:
            assert isinstance(manipulation_type.value, str)
            # Values should be lowercase and use underscores or be single words
            assert manipulation_type.value.islower()


class TestManipulationResult:
    """Test cases for ManipulationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a ManipulationResult."""
        result = ManipulationResult(
            manipulation_type=ManipulationType.CLICKBAIT,
            confidence=0.8,
            matched_text="You won't believe",
            description="Uses sensational language",
            suggestions=["Look for factual headlines"]
        )
        
        assert result.manipulation_type == ManipulationType.CLICKBAIT
        assert result.confidence == 0.8
        assert result.matched_text == "You won't believe"
        assert result.description == "Uses sensational language"
        assert len(result.suggestions) == 1


if __name__ == "__main__":
    pytest.main([__file__])
