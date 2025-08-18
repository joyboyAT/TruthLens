"""
Tests for LLM manipulation detection module.
"""

import pytest
from unittest.mock import Mock, patch
from src.manipulation.llm_detector import (
    LLMManipulationDetector,
    LLMProvider,
    LLMDetectionResult
)


class TestLLMManipulationDetector:
    """Test cases for LLMManipulationDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LLMManipulationDetector(provider=LLMProvider.OPENAI)
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector is not None
        assert self.detector.provider == LLMProvider.OPENAI
        assert self.detector.model_name == "gpt-3.5-turbo"
    
    def test_fallback_detection(self):
        """Test fallback detection when LLM is not available."""
        text = "You won't believe this shocking conspiracy!"
        result = self.detector._fallback_detection(text)
        
        assert isinstance(result, LLMDetectionResult)
        assert hasattr(result, 'manipulation_types')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'suggestions')
    
    def test_fallback_detection_with_clickbait(self):
        """Test fallback detection with clickbait keywords."""
        text = "This is shocking and amazing news!"
        result = self.detector._fallback_detection(text)
        
        assert 'clickbait' in result.manipulation_types
    
    def test_fallback_detection_with_fear_appeal(self):
        """Test fallback detection with fear appeal keywords."""
        text = "This dangerous chemical could kill you!"
        result = self.detector._fallback_detection(text)
        
        assert 'fear_appeal' in result.manipulation_types
    
    def test_fallback_detection_with_conspiracy(self):
        """Test fallback detection with conspiracy keywords."""
        text = "The mainstream media is hiding this conspiracy!"
        result = self.detector._fallback_detection(text)
        
        assert 'conspiracy' in result.manipulation_types
    
    def test_fallback_detection_no_manipulation(self):
        """Test fallback detection with no manipulation keywords."""
        text = "This is a factual statement about the weather."
        result = self.detector._fallback_detection(text)
        
        assert len(result.manipulation_types) == 0
        assert result.confidence == 0.5
    
    @patch('src.manipulation.llm_detector.openai')
    def test_openai_detection_success(self, mock_openai):
        """Test successful OpenAI detection."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "manipulation_types": ["clickbait", "fear_appeal"],
            "confidence": 0.85,
            "reasoning": "Text uses sensational language and fear-based claims",
            "specific_examples": ["shocking", "dangerous"],
            "suggestions": ["Look for factual headlines"],
            "bias_indicators": ["emotional language"],
            "logical_fallacies": ["appeal to emotion"]
        }
        '''
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        text = "You won't believe this shocking and dangerous conspiracy!"
        result = self.detector._detect_with_openai(text)
        
        assert isinstance(result, LLMDetectionResult)
        assert "clickbait" in result.manipulation_types
        assert "fear_appeal" in result.manipulation_types
        assert result.confidence == 0.85
    
    @patch('src.manipulation.llm_detector.openai')
    def test_openai_detection_failure(self, mock_openai):
        """Test OpenAI detection failure falls back to rule-based."""
        mock_openai.ChatCompletion.create.side_effect = Exception("API Error")
        
        text = "You won't believe this shocking conspiracy!"
        result = self.detector._detect_with_openai(text)
        
        # Should fall back to rule-based detection
        assert isinstance(result, LLMDetectionResult)
        assert result.confidence == 0.5
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        response_text = '''
        {
            "manipulation_types": ["clickbait"],
            "confidence": 0.8,
            "reasoning": "Uses sensational language",
            "specific_examples": ["shocking"],
            "suggestions": ["Look for facts"],
            "bias_indicators": [],
            "logical_fallacies": []
        }
        '''
        
        result = self.detector._parse_llm_response(response_text)
        
        assert isinstance(result, LLMDetectionResult)
        assert "clickbait" in result.manipulation_types
        assert result.confidence == 0.8
        assert result.reasoning == "Uses sensational language"
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        response_text = "This is not valid JSON"
        
        result = self.detector._parse_llm_response(response_text)
        
        # Should fall back to rule-based detection
        assert isinstance(result, LLMDetectionResult)
        assert result.confidence == 0.5
    
    def test_parse_llm_response_missing_json(self):
        """Test parsing response with no JSON."""
        response_text = "Here is my analysis: The text shows manipulation."
        
        result = self.detector._parse_llm_response(response_text)
        
        # Should fall back to rule-based detection
        assert isinstance(result, LLMDetectionResult)
        assert result.confidence == 0.5
    
    def test_build_analysis_prompt(self):
        """Test building analysis prompt."""
        text = "Test text for analysis"
        prompt = self.detector._build_analysis_prompt(text)
        
        assert text in prompt
        assert "TEXT TO ANALYZE:" in prompt
        assert "JSON format" in prompt
    
    def test_build_analysis_prompt_with_claim(self):
        """Test building analysis prompt with claim."""
        text = "Test text for analysis"
        claim = "Specific claim to focus on"
        prompt = self.detector._build_analysis_prompt(text, claim)
        
        assert text in prompt
        assert claim in prompt
        assert "FOCUS CLAIM:" in prompt
    
    def test_get_system_prompt(self):
        """Test getting system prompt."""
        prompt = self.detector._get_system_prompt()
        
        assert "expert fact-checker" in prompt
        assert "manipulation techniques" in prompt
        assert "JSON format" in prompt
        assert "manipulation_types" in prompt


class TestLLMProvider:
    """Test cases for LLMProvider enum."""
    
    def test_enum_values(self):
        """Test that all providers are defined."""
        expected_providers = {'openai', 'vertex_ai', 'local'}
        actual_providers = {p.value for p in LLMProvider}
        assert actual_providers == expected_providers


class TestLLMDetectionResult:
    """Test cases for LLMDetectionResult dataclass."""
    
    def test_result_creation(self):
        """Test creating an LLMDetectionResult."""
        result = LLMDetectionResult(
            manipulation_types=["clickbait"],
            confidence=0.8,
            reasoning="Uses sensational language",
            specific_examples=["shocking"],
            suggestions=["Look for facts"],
            bias_indicators=[],
            logical_fallacies=[]
        )
        
        assert result.manipulation_types == ["clickbait"]
        assert result.confidence == 0.8
        assert result.reasoning == "Uses sensational language"
        assert result.specific_examples == ["shocking"]
        assert result.suggestions == ["Look for facts"]


class TestBiasAnalysis:
    """Test cases for bias analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LLMManipulationDetector(provider=LLMProvider.OPENAI)
    
    @patch('src.manipulation.llm_detector.openai')
    def test_analyze_bias_success(self, mock_openai):
        """Test successful bias analysis."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "bias_types": ["confirmation_bias"],
            "confidence": 0.7,
            "explanation": "Text shows confirmation bias",
            "examples": ["selective evidence"],
            "suggestions": ["Check multiple sources"]
        }
        '''
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        text = "This proves my point because I found evidence that supports it."
        result = self.detector.analyze_bias(text)
        
        assert result["bias_types"] == ["confirmation_bias"]
        assert result["confidence"] == 0.7
    
    def test_analyze_bias_fallback(self):
        """Test bias analysis fallback when LLM not available."""
        text = "Test text for bias analysis"
        result = self.detector.analyze_bias(text)
        
        assert "bias_types" in result
        assert "confidence" in result
        assert "explanation" in result
        assert result["confidence"] == 0.0


class TestLogicalFallacyDetection:
    """Test cases for logical fallacy detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = LLMManipulationDetector(provider=LLMProvider.OPENAI)
    
    @patch('src.manipulation.llm_detector.openai')
    def test_detect_logical_fallacies_success(self, mock_openai):
        """Test successful logical fallacy detection."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "fallacies": [
                {
                    "type": "appeal_to_emotion",
                    "description": "Uses emotional language",
                    "example": "This is heartbreaking",
                    "confidence": 0.8
                }
            ]
        }
        '''
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        text = "This is heartbreaking and you should feel terrible about it."
        result = self.detector.detect_logical_fallacies(text)
        
        assert len(result) == 1
        assert result[0]["type"] == "appeal_to_emotion"
        assert result[0]["confidence"] == 0.8
    
    def test_detect_logical_fallacies_fallback(self):
        """Test logical fallacy detection fallback."""
        text = "Test text for fallacy detection"
        result = self.detector.detect_logical_fallacies(text)
        
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__])
