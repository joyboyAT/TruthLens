"""
LLM-based manipulation detection for Phase 5.

Uses AI models to detect subtle manipulation patterns that rule-based
detection might miss, including:
- Contextual manipulation
- Logical fallacies
- Bias detection
- Emotional manipulation
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    import openai
except ImportError:
    openai = None


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    VERTEX_AI = "vertex_ai"
    LOCAL = "local"


@dataclass
class LLMDetectionResult:
    """Result of LLM-based manipulation detection."""
    manipulation_types: List[str]
    confidence: float
    reasoning: str
    specific_examples: List[str]
    suggestions: List[str]
    bias_indicators: List[str]
    logical_fallacies: List[str]


class LLMManipulationDetector:
    """LLM-based detector for subtle manipulation patterns."""
    
    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = "gpt-3.5-turbo"  # Default model
        
        if provider == LLMProvider.OPENAI and openai:
            if api_key:
                openai.api_key = api_key
            else:
                # Try to get from environment
                import os
                openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def detect_manipulation(self, text: str, claim: Optional[str] = None) -> LLMDetectionResult:
        """
        Detect manipulation using LLM analysis.
        
        Args:
            text: Input text to analyze
            claim: Optional specific claim to focus on
            
        Returns:
            LLMDetectionResult with detailed analysis
        """
        if self.provider == LLMProvider.OPENAI and openai:
            return self._detect_with_openai(text, claim)
        else:
            # Fallback to rule-based detection
            return self._fallback_detection(text, claim)
    
    def _detect_with_openai(self, text: str, claim: Optional[str] = None) -> LLMDetectionResult:
        """Detect manipulation using OpenAI API."""
        
        prompt = self._build_analysis_prompt(text, claim)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            return self._parse_llm_response(result_text)
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._fallback_detection(text, claim)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for manipulation detection."""
        return """You are an expert fact-checker and manipulation detection specialist. 
Your task is to analyze text for various types of manipulation techniques and provide detailed analysis.

Focus on detecting:
1. Clickbait and sensationalism
2. Fear appeal and emotional manipulation
3. Cherry-picking and selective evidence
4. False authority and appeal to authority
5. Missing context and incomplete information
6. Logical fallacies
7. Bias indicators
8. Conspiracy language
9. Urgency and pressure tactics

Provide your analysis in JSON format with the following structure:
{
    "manipulation_types": ["list", "of", "detected", "types"],
    "confidence": 0.85,
    "reasoning": "Detailed explanation of why manipulation was detected",
    "specific_examples": ["specific", "examples", "from", "text"],
    "suggestions": ["suggestions", "for", "addressing", "manipulation"],
    "bias_indicators": ["indicators", "of", "bias"],
    "logical_fallacies": ["logical", "fallacies", "detected"]
}"""
    
    def _build_analysis_prompt(self, text: str, claim: Optional[str] = None) -> str:
        """Build the analysis prompt for the LLM."""
        prompt = f"""Analyze the following text for manipulation techniques:

TEXT TO ANALYZE:
{text}

"""
        
        if claim:
            prompt += f"FOCUS CLAIM: {claim}\n\n"
        
        prompt += """Please provide a detailed analysis of any manipulation techniques found, 
including specific examples from the text and suggestions for addressing them.

Respond in JSON format as specified in the system prompt."""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> LLMDetectionResult:
        """Parse the LLM response into a structured result."""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return LLMDetectionResult(
                    manipulation_types=data.get('manipulation_types', []),
                    confidence=data.get('confidence', 0.0),
                    reasoning=data.get('reasoning', ''),
                    specific_examples=data.get('specific_examples', []),
                    suggestions=data.get('suggestions', []),
                    bias_indicators=data.get('bias_indicators', []),
                    logical_fallacies=data.get('logical_fallacies', [])
                )
            else:
                return self._fallback_detection(response_text)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            return self._fallback_detection(response_text)
    
    def _fallback_detection(self, text: str, claim: Optional[str] = None) -> LLMDetectionResult:
        """Fallback detection when LLM is not available."""
        # Basic fallback analysis
        manipulation_types = []
        confidence = 0.5
        reasoning = "Basic analysis performed (LLM not available)"
        specific_examples = []
        suggestions = [
            "Look for evidence-based information",
            "Check multiple credible sources",
            "Verify claims with fact-checking organizations"
        ]
        bias_indicators = []
        logical_fallacies = []
        
        # Simple keyword-based detection
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['shocking', 'amazing', 'incredible']):
            manipulation_types.append('clickbait')
        
        if any(word in text_lower for word in ['dangerous', 'deadly', 'threat']):
            manipulation_types.append('fear_appeal')
        
        if any(word in text_lower for word in ['conspiracy', 'cover-up', 'hidden']):
            manipulation_types.append('conspiracy')
        
        return LLMDetectionResult(
            manipulation_types=manipulation_types,
            confidence=confidence,
            reasoning=reasoning,
            specific_examples=specific_examples,
            suggestions=suggestions,
            bias_indicators=bias_indicators,
            logical_fallacies=logical_fallacies
        )
    
    def analyze_bias(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for bias indicators.
        
        Args:
            text: Text to analyze for bias
            
        Returns:
            Dictionary with bias analysis results
        """
        prompt = f"""Analyze the following text for bias indicators:

TEXT:
{text}

Identify any of these types of bias:
1. Confirmation bias
2. Selection bias
3. Framing bias
4. Authority bias
5. Emotional bias
6. Cultural bias
7. Political bias

Provide analysis in JSON format:
{{
    "bias_types": ["list", "of", "bias", "types"],
    "confidence": 0.8,
    "explanation": "Explanation of bias detected",
    "examples": ["specific", "examples"],
    "suggestions": ["how", "to", "address", "bias"]
}}"""

        try:
            if self.provider == LLMProvider.OPENAI and openai:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a bias detection expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                result_text = response.choices[0].message.content
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    json_str = result_text[json_start:json_end]
                    return json.loads(json_str)
            
        except Exception as e:
            print(f"Bias analysis error: {e}")
        
        # Fallback
        return {
            "bias_types": [],
            "confidence": 0.0,
            "explanation": "Bias analysis not available",
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
        prompt = f"""Analyze the following text for logical fallacies:

TEXT:
{text}

Common logical fallacies to look for:
1. Ad hominem
2. Straw man
3. False dilemma
4. Appeal to authority
5. Appeal to emotion
6. Hasty generalization
7. Post hoc ergo propter hoc
8. Slippery slope
9. Circular reasoning
10. Red herring

Provide analysis in JSON format:
{{
    "fallacies": [
        {{
            "type": "fallacy_type",
            "description": "Description of the fallacy",
            "example": "Specific example from text",
            "confidence": 0.8
        }}
    ]
}}"""

        try:
            if self.provider == LLMProvider.OPENAI and openai:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a logical fallacy detection expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                result_text = response.choices[0].message.content
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    json_str = result_text[json_start:json_end]
                    data = json.loads(json_str)
                    return data.get('fallacies', [])
            
        except Exception as e:
            print(f"Logical fallacy detection error: {e}")
        
        # Fallback
        return []
