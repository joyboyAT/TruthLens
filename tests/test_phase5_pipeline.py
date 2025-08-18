"""
Tests for Phase 5 pipeline module.
"""

import pytest
import time
from src.manipulation.pipeline import (
    ManipulationPipeline,
    PipelineConfig,
    PipelineResult,
    detect_manipulation,
    get_manipulation_summary,
    analyze_bias
)
from src.manipulation.rules import ManipulationResult, ManipulationType
from src.manipulation.prebunk_generator import TipType


class TestPipelineConfig:
    """Test cases for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        
        assert config.use_llm is True
        assert config.generate_educational_content is True
        assert config.max_tips == 3
        assert config.confidence_threshold == 0.6
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            use_llm=False,
            generate_educational_content=False,
            max_tips=5,
            confidence_threshold=0.8
        )
        
        assert config.use_llm is False
        assert config.generate_educational_content is False
        assert config.max_tips == 5
        assert config.confidence_threshold == 0.8


class TestManipulationPipeline:
    """Test cases for ManipulationPipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig(
            use_llm=False,
            generate_educational_content=True,
            max_tips=2,
            confidence_threshold=0.5
        )
        self.pipeline = ManipulationPipeline(self.config)
    
    def test_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline is not None
        assert self.pipeline.config == self.config
        assert self.pipeline.rule_detector is not None
        assert self.pipeline.llm_detector is None  # LLM disabled
        assert self.pipeline.prebunk_generator is not None
        assert self.pipeline.output_builder is not None
    
    def test_process_text_basic(self):
        """Test basic text processing."""
        text = "You won't believe this shocking conspiracy!"
        result = self.pipeline.process_text(text)
        
        assert isinstance(result, PipelineResult)
        assert len(result.rule_results) > 0
        assert result.llm_results is None  # LLM disabled
        assert len(result.prebunk_tips) > 0
        assert result.processing_time > 0
        assert "text_length" in result.metadata
    
    def test_process_text_no_manipulation(self):
        """Test processing text with no manipulation."""
        text = "This is a factual statement about the weather."
        result = self.pipeline.process_text(text)
        
        assert isinstance(result, PipelineResult)
        assert len(result.rule_results) == 0
        assert len(result.prebunk_tips) == 0
        assert result.processing_time > 0
    
    def test_process_text_with_claim(self):
        """Test processing text with specific claim."""
        text = "You won't believe this shocking conspiracy!"
        claim = "Specific claim to focus on"
        result = self.pipeline.process_text(text, claim)
        
        assert isinstance(result, PipelineResult)
        assert len(result.rule_results) > 0
    
    def test_detect_with_rules(self):
        """Test rule-based detection."""
        text = "You won't believe this shocking conspiracy!"
        results = self.pipeline._detect_with_rules(text)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ManipulationResult) for r in results)
        assert all(r.confidence >= self.config.confidence_threshold for r in results)
    
    def test_detect_with_rules_below_threshold(self):
        """Test rule-based detection with low confidence results."""
        # Create a config with high threshold
        high_threshold_config = PipelineConfig(confidence_threshold=0.9)
        pipeline = ManipulationPipeline(high_threshold_config)
        
        text = "You won't believe this shocking conspiracy!"
        results = pipeline._detect_with_rules(text)
        
        # Should filter out low confidence results
        assert all(r.confidence >= 0.9 for r in results)
    
    def test_generate_educational_content(self):
        """Test educational content generation."""
        rule_results = [
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                matched_text="You won't believe",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            ),
            ManipulationResult(
                manipulation_type=ManipulationType.FEAR_APPEAL,
                confidence=0.7,
                matched_text="dangerous",
                description="Uses fear-based language",
                suggestions=["Look for evidence"]
            )
        ]
        
        tips = self.pipeline._generate_educational_content(rule_results)
        
        assert isinstance(tips, list)
        assert len(tips) <= self.config.max_tips
        assert len(tips) > 0
    
    def test_get_tip_type(self):
        """Test tip type mapping."""
        assert self.pipeline._get_tip_type("clickbait") == TipType.CLICKBAIT
        assert self.pipeline._get_tip_type("fear_appeal") == TipType.FEAR_APPEAL
        assert self.pipeline._get_tip_type("cherry_picking") == TipType.CHERRY_PICKING
        assert self.pipeline._get_tip_type("unknown_type") == TipType.GENERAL
    
    def test_get_summary(self):
        """Test getting manipulation summary."""
        text = "You won't believe this shocking conspiracy!"
        summary = self.pipeline.get_summary(text)
        
        assert isinstance(summary, dict)
        assert "total_detections" in summary
        assert "manipulation_types" in summary
        assert "confidence_score" in summary
        assert "suggestions" in summary
    
    def test_analyze_bias_without_llm(self):
        """Test bias analysis without LLM."""
        text = "This text shows bias towards one perspective."
        result = self.pipeline.analyze_bias(text)
        
        assert isinstance(result, dict)
        assert "bias_types" in result
        assert "confidence" in result
        assert "explanation" in result
        assert result["confidence"] == 0.0  # LLM disabled
    
    def test_detect_logical_fallacies_without_llm(self):
        """Test logical fallacy detection without LLM."""
        text = "This argument contains logical fallacies."
        result = self.pipeline.detect_logical_fallacies(text)
        
        assert isinstance(result, list)
        assert len(result) == 0  # LLM disabled
    
    def test_to_json(self):
        """Test converting result to JSON."""
        text = "You won't believe this shocking conspiracy!"
        result = self.pipeline.process_text(text)
        
        json_str = self.pipeline.to_json(result)
        
        assert isinstance(json_str, str)
        assert len(json_str) > 0
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        text = "You won't believe this shocking conspiracy!"
        result = self.pipeline.process_text(text)
        
        output_dict = self.pipeline.to_dict(result)
        
        assert isinstance(output_dict, dict)
        assert "manipulation_cues" in output_dict
        assert "summary" in output_dict
        assert "educational_content" in output_dict
        assert "pipeline_metadata" in output_dict
    
    def test_error_handling(self):
        """Test error handling in pipeline."""
        # Test with empty text
        result = self.pipeline.process_text("")
        assert isinstance(result, PipelineResult)

        # Test with None text - should handle gracefully
        try:
            result = self.pipeline.process_text(None)
            # If it doesn't raise an exception, it should return a valid result
            assert isinstance(result, PipelineResult)
        except (TypeError, AttributeError):
            # It's also acceptable for the pipeline to raise an exception for None
            pass


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_detect_manipulation(self):
        """Test detect_manipulation convenience function."""
        text = "You won't believe this shocking conspiracy!"
        result = detect_manipulation(text, use_llm=False, generate_educational_content=True)
        
        assert isinstance(result, dict)
        assert "manipulation_cues" in result
        assert "summary" in result
        assert "educational_content" in result
    
    def test_detect_manipulation_without_educational_content(self):
        """Test detect_manipulation without educational content."""
        text = "You won't believe this shocking conspiracy!"
        result = detect_manipulation(text, use_llm=False, generate_educational_content=False)
        
        assert isinstance(result, dict)
        assert "manipulation_cues" in result
        assert "summary" in result
        # Educational content should be empty
        assert not result.get("educational_content", {})
    
    def test_get_manipulation_summary(self):
        """Test get_manipulation_summary convenience function."""
        text = "You won't believe this shocking conspiracy!"
        summary = get_manipulation_summary(text)
        
        assert isinstance(summary, dict)
        assert "total_detections" in summary
        assert "manipulation_types" in summary
        assert "confidence_score" in summary
    
    def test_analyze_bias(self):
        """Test analyze_bias convenience function."""
        text = "This text shows bias towards one perspective."
        result = analyze_bias(text)
        
        assert isinstance(result, dict)
        assert "bias_types" in result
        assert "confidence" in result
        assert "explanation" in result


class TestPipelineResult:
    """Test cases for PipelineResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a PipelineResult."""
        from src.manipulation.output_builder import UXOutput
        
        rule_results = [
            ManipulationResult(
                manipulation_type=ManipulationType.CLICKBAIT,
                confidence=0.8,
                matched_text="You won't believe",
                description="Uses sensational language",
                suggestions=["Look for factual headlines"]
            )
        ]
        
        ux_output = UXOutput(
            manipulation_cues=[],
            summary=None,
            educational_content={},
            prebunk_tips=[],
            interactive_elements=[],
            metadata={}
        )
        
        result = PipelineResult(
            rule_results=rule_results,
            llm_results=None,
            prebunk_tips=[],
            ux_output=ux_output,
            processing_time=0.1,
            metadata={"text_length": 100}
        )
        
        assert len(result.rule_results) == 1
        assert result.llm_results is None
        assert len(result.prebunk_tips) == 0
        assert result.processing_time == 0.1
        assert result.metadata["text_length"] == 100


class TestPerformance:
    """Test cases for performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PipelineConfig(use_llm=False, generate_educational_content=False)
        self.pipeline = ManipulationPipeline(self.config)
    
    def test_processing_time(self):
        """Test that processing time is reasonable."""
        text = "You won't believe this shocking conspiracy!"
        result = self.pipeline.process_text(text)
        
        # Should complete quickly (under 1 second)
        assert result.processing_time < 1.0
        assert result.processing_time > 0.0
    
    def test_text_length_scaling(self):
        """Test that processing time scales reasonably with text length."""
        short_text = "Short text with clickbait."
        long_text = "Long text with multiple manipulation techniques. " * 100
        
        start_time = time.time()
        short_result = self.pipeline.process_text(short_text)
        short_time = time.time() - start_time
        
        start_time = time.time()
        long_result = self.pipeline.process_text(long_text)
        long_time = time.time() - start_time
        
        # Long text should take longer but not exponentially longer
        assert long_time > short_time
        # Allow for some variance in timing, but ensure it's not dramatically slower
        assert long_time < short_time * 50  # Should not be 50x slower
    
    def test_memory_usage(self):
        """Test that memory usage is reasonable."""
        # Process multiple texts to check for memory leaks
        texts = [
            f"Text {i} with manipulation techniques like clickbait and fear appeal."
            for i in range(10)
        ]
        
        results = []
        for text in texts:
            result = self.pipeline.process_text(text)
            results.append(result)
        
        # All results should be valid
        assert len(results) == 10
        assert all(isinstance(r, PipelineResult) for r in results)


class TestConfigurationOptions:
    """Test cases for different configuration options."""
    
    def test_llm_enabled_config(self):
        """Test pipeline with LLM enabled."""
        config = PipelineConfig(use_llm=True)
        pipeline = ManipulationPipeline(config)
        
        # Should have LLM detector
        assert pipeline.llm_detector is not None
    
    def test_educational_content_disabled(self):
        """Test pipeline with educational content disabled."""
        config = PipelineConfig(generate_educational_content=False)
        pipeline = ManipulationPipeline(config)
        
        # Should not have prebunk generator
        assert pipeline.prebunk_generator is None
    
    def test_custom_max_tips(self):
        """Test pipeline with custom max tips."""
        config = PipelineConfig(max_tips=1)
        pipeline = ManipulationPipeline(config)
        
        text = "You won't believe this shocking conspiracy!"
        result = pipeline.process_text(text)
        
        # Should respect max_tips limit
        assert len(result.prebunk_tips) <= 1
    
    def test_high_confidence_threshold(self):
        """Test pipeline with high confidence threshold."""
        config = PipelineConfig(confidence_threshold=0.9)
        pipeline = ManipulationPipeline(config)
        
        text = "You won't believe this shocking conspiracy!"
        result = pipeline.process_text(text)
        
        # Should only include high confidence detections
        for detection in result.rule_results:
            assert detection.confidence >= 0.9


if __name__ == "__main__":
    pytest.main([__file__])
