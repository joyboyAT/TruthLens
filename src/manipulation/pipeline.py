"""
End-to-end Phase 5 pipeline for manipulation detection and user education.

Orchestrates the complete workflow:
1. Rule-based manipulation detection
2. LLM-based pattern recognition
3. Prebunk tip generation
4. UX output formatting
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .rules import ManipulationRuleDetector, ManipulationResult
from .llm_detector import LLMManipulationDetector, LLMProvider
from .prebunk_generator import PrebunkGenerator, TipType
from .output_builder import ManipulationOutputBuilder, UXOutput


@dataclass
class PipelineConfig:
    """Configuration for the manipulation detection pipeline."""
    use_llm: bool = True
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_api_key: Optional[str] = None
    generate_educational_content: bool = True
    max_tips: int = 3
    confidence_threshold: float = 0.6


@dataclass
class PipelineResult:
    """Result of the complete manipulation detection pipeline."""
    rule_results: List[ManipulationResult]
    prebunk_tips: List[Any]
    ux_output: UXOutput
    processing_time: float
    metadata: Dict[str, Any]
    llm_results: Optional[Any] = None


class ManipulationPipeline:
    """End-to-end pipeline for manipulation detection and user education."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.rule_detector = ManipulationRuleDetector()
        self.llm_detector = None
        self.prebunk_generator = None
        self.output_builder = ManipulationOutputBuilder()
        
        # Initialize LLM detector if enabled
        if self.config.use_llm:
            self.llm_detector = LLMManipulationDetector(
                provider=self.config.llm_provider,
                api_key=self.config.llm_api_key
            )
        
        # Initialize prebunk generator if enabled
        if self.config.generate_educational_content:
            self.prebunk_generator = PrebunkGenerator(
                use_llm=self.config.use_llm,
                api_key=self.config.llm_api_key
            )
    
    def process_text(self, text: str, claim: Optional[str] = None) -> PipelineResult:
        """
        Process text through the complete manipulation detection pipeline.
        
        Args:
            text: Input text to analyze
            claim: Optional specific claim to focus on
            
        Returns:
            PipelineResult with complete analysis
        """
        start_time = time.time()
        
        # Step 1: Rule-based detection
        rule_results = self._detect_with_rules(text)
        
        # Step 2: LLM-based detection (if enabled)
        llm_results = None
        if self.config.use_llm and self.llm_detector:
            llm_results = self._detect_with_llm(text, claim)
        
        # Step 3: Generate educational content (if enabled)
        prebunk_tips = []
        if self.config.generate_educational_content and self.prebunk_generator:
            prebunk_tips = self._generate_educational_content(rule_results, llm_results)
        
        # Step 4: Build UX output
        ux_output = self._build_ux_output(rule_results, llm_results, prebunk_tips)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build metadata
        metadata = {
            "text_length": len(text),
            "detection_methods": ["rule_based"],
            "config": {
                "use_llm": self.config.use_llm,
                "generate_educational_content": self.config.generate_educational_content,
                "confidence_threshold": self.config.confidence_threshold
            }
        }
        
        if llm_results:
            metadata["detection_methods"].append("llm_based")
        
        return PipelineResult(
            rule_results=rule_results,
            llm_results=llm_results,
            prebunk_tips=prebunk_tips,
            ux_output=ux_output,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def _detect_with_rules(self, text: str) -> List[ManipulationResult]:
        """Perform rule-based manipulation detection."""
        try:
            results = self.rule_detector.detect_manipulation(text)
            
            # Filter by confidence threshold
            filtered_results = [
                result for result in results 
                if result.confidence >= self.config.confidence_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            print(f"Error in rule-based detection: {e}")
            return []
    
    def _detect_with_llm(self, text: str, claim: Optional[str] = None) -> Optional[Any]:
        """Perform LLM-based manipulation detection."""
        try:
            if self.llm_detector:
                return self.llm_detector.detect_manipulation(text, claim)
        except Exception as e:
            print(f"Error in LLM-based detection: {e}")
        
        return None
    
    def _generate_educational_content(self, rule_results: List[ManipulationResult], 
                                    llm_results: Optional[Any] = None) -> List[Any]:
        """Generate educational content based on detected manipulation."""
        if not self.prebunk_generator:
            return []
        
        # Collect unique manipulation types
        manipulation_types = set()
        
        # From rule results
        for result in rule_results:
            manipulation_types.add(result.manipulation_type.value)
        
        # From LLM results
        if llm_results and hasattr(llm_results, 'manipulation_types'):
            for manipulation_type in llm_results.manipulation_types:
                manipulation_types.add(manipulation_type)
        
        # Generate tips for each manipulation type
        tips = []
        for manipulation_type in list(manipulation_types)[:self.config.max_tips]:
            try:
                tip_type = self._get_tip_type(manipulation_type)
                tip = self.prebunk_generator.generate_tip(tip_type)
                tips.append(tip)
            except Exception as e:
                print(f"Error generating tip for {manipulation_type}: {e}")
        
        return tips
    
    def _get_tip_type(self, manipulation_type: str) -> TipType:
        """Convert manipulation type string to TipType enum."""
        type_mapping = {
            "clickbait": TipType.CLICKBAIT,
            "fear_appeal": TipType.FEAR_APPEAL,
            "cherry_picking": TipType.CHERRY_PICKING,
            "false_authority": TipType.FALSE_AUTHORITY,
            "missing_context": TipType.MISSING_CONTEXT,
            "emotional_manipulation": TipType.EMOTIONAL_MANIPULATION,
            "urgency": TipType.URGENCY,
            "conspiracy": TipType.CONSPIRACY
        }
        
        return type_mapping.get(manipulation_type, TipType.GENERAL)
    
    def _build_ux_output(self, rule_results: List[ManipulationResult],
                        llm_results: Optional[Any] = None,
                        prebunk_tips: Optional[List[Any]] = None) -> UXOutput:
        """Build UX output from all detection results."""
        return self.output_builder.build_complete_output(
            rule_results=rule_results,
            llm_results=llm_results,
            prebunk_tips=prebunk_tips
        )
    
    def get_summary(self, text: str) -> Dict[str, Any]:
        """
        Get a quick summary of manipulation detection without full processing.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Summary dictionary
        """
        rule_results = self._detect_with_rules(text)
        
        return self.rule_detector.get_manipulation_summary(text)
    
    def analyze_bias(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for bias indicators.
        
        Args:
            text: Text to analyze for bias
            
        Returns:
            Bias analysis results
        """
        if self.llm_detector:
            return self.llm_detector.analyze_bias(text)
        
        return {
            "bias_types": [],
            "confidence": 0.0,
            "explanation": "Bias analysis not available (LLM disabled)",
            "examples": [],
            "suggestions": ["Check multiple perspectives", "Verify sources"]
        }
    
    def detect_logical_fallacies(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect logical fallacies in text.
        
        Args:
            text: Text to analyze for logical fallacies
            
        Returns:
            List of detected logical fallacies
        """
        if self.llm_detector:
            return self.llm_detector.detect_logical_fallacies(text)
        
        return []
    
    def to_json(self, result: PipelineResult) -> str:
        """
        Convert pipeline result to JSON string.
        
        Args:
            result: PipelineResult to convert
            
        Returns:
            JSON string representation
        """
        return self.output_builder.to_json(result.ux_output)
    
    def to_dict(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Convert pipeline result to dictionary.
        
        Args:
            result: PipelineResult to convert
            
        Returns:
            Dictionary representation
        """
        output_dict = self.output_builder.to_dict(result.ux_output)
        
        # Add pipeline-specific metadata
        output_dict["pipeline_metadata"] = {
            "processing_time": result.processing_time,
            "rule_detections": len(result.rule_results),
            "llm_detections": len(result.llm_results.manipulation_types) if result.llm_results else 0,
            "educational_tips": len(result.prebunk_tips),
            "config": result.metadata["config"]
        }
        
        return output_dict


# Convenience functions for easy usage
def detect_manipulation(text: str, 
                       use_llm: bool = True,
                       generate_educational_content: bool = True,
                       api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to detect manipulation in text.
    
    Args:
        text: Text to analyze
        use_llm: Whether to use LLM-based detection
        generate_educational_content: Whether to generate educational content
        api_key: Optional API key for LLM services
        
    Returns:
        Dictionary with manipulation detection results
    """
    config = PipelineConfig(
        use_llm=use_llm,
        generate_educational_content=generate_educational_content,
        llm_api_key=api_key
    )
    
    pipeline = ManipulationPipeline(config)
    result = pipeline.process_text(text)
    
    return pipeline.to_dict(result)


def get_manipulation_summary(text: str) -> Dict[str, Any]:
    """
    Get a quick summary of manipulation detection.
    
    Args:
        text: Text to analyze
        
    Returns:
        Summary dictionary
    """
    pipeline = ManipulationPipeline()
    return pipeline.get_summary(text)


def analyze_bias(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze text for bias indicators.
    
    Args:
        text: Text to analyze
        api_key: Optional API key for LLM services
        
    Returns:
        Bias analysis results
    """
    config = PipelineConfig(use_llm=True, llm_api_key=api_key)
    pipeline = ManipulationPipeline(config)
    return pipeline.analyze_bias(text)
