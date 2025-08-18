"""
Prebunk tip generator for Phase 5.

Generates educational content to help users understand and recognize
manipulation techniques, including:
- Prebunking tips
- Educational cards
- Manipulation pattern explanations
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    import openai
except ImportError:
    openai = None


class TipType(Enum):
    """Types of prebunking tips."""
    CLICKBAIT = "clickbait"
    FEAR_APPEAL = "fear_appeal"
    CHERRY_PICKING = "cherry_picking"
    FALSE_AUTHORITY = "false_authority"
    MISSING_CONTEXT = "missing_context"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    URGENCY = "urgency"
    CONSPIRACY = "conspiracy"
    GENERAL = "general"


@dataclass
class PrebunkTip:
    """A prebunking educational tip."""
    tip_type: TipType
    title: str
    description: str
    examples: List[str]
    red_flags: List[str]
    how_to_spot: List[str]
    what_to_do: List[str]
    difficulty: str  # "easy", "medium", "hard"
    estimated_read_time: int  # in seconds


@dataclass
class EducationalCard:
    """An educational card for user interface."""
    card_id: str
    title: str
    content: str
    visual_elements: List[str]
    interactive_elements: List[str]
    difficulty: str
    estimated_time: int


class PrebunkGenerator:
    """Generator for prebunking educational content."""
    
    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None):
        self.use_llm = use_llm and openai is not None
        self.api_key = api_key
        
        if self.use_llm and api_key:
            openai.api_key = api_key
        elif self.use_llm:
            import os
            openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def generate_tip(self, manipulation_type: TipType, context: Optional[str] = None) -> PrebunkTip:
        """
        Generate a prebunking tip for a specific manipulation type.
        
        Args:
            manipulation_type: Type of manipulation to create tip for
            context: Optional context about the specific content
            
        Returns:
            PrebunkTip with educational content
        """
        if self.use_llm:
            return self._generate_with_llm(manipulation_type, context)
        else:
            return self._generate_template_tip(manipulation_type)
    
    def _generate_with_llm(self, manipulation_type: TipType, context: Optional[str] = None) -> PrebunkTip:
        """Generate tip using LLM."""
        
        prompt = self._build_tip_prompt(manipulation_type, context)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self._get_tip_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content
            return self._parse_tip_response(result_text, manipulation_type)
            
        except Exception as e:
            print(f"LLM tip generation error: {e}")
            return self._generate_template_tip(manipulation_type)
    
    def _get_tip_system_prompt(self) -> str:
        """Get system prompt for tip generation."""
        return """You are an expert in media literacy and fact-checking education. 
Your task is to create engaging, educational prebunking tips that help users recognize manipulation techniques.

Create tips that are:
1. Clear and easy to understand
2. Practical and actionable
3. Engaging and memorable
4. Based on real-world examples

Provide your response in JSON format:
{
    "title": "Catchy title for the tip",
    "description": "Clear explanation of the manipulation technique",
    "examples": ["real-world", "examples"],
    "red_flags": ["warning", "signs"],
    "how_to_spot": ["ways", "to", "identify"],
    "what_to_do": ["actions", "to", "take"],
    "difficulty": "easy|medium|hard",
    "estimated_read_time": 30
}"""
    
    def _build_tip_prompt(self, manipulation_type: TipType, context: Optional[str] = None) -> str:
        """Build prompt for tip generation."""
        
        manipulation_descriptions = {
            TipType.CLICKBAIT: "clickbait headlines and sensational language",
            TipType.FEAR_APPEAL: "fear-based manipulation and scare tactics",
            TipType.CHERRY_PICKING: "selective evidence and cherry-picking",
            TipType.FALSE_AUTHORITY: "false authority and appeal to authority",
            TipType.MISSING_CONTEXT: "missing context and incomplete information",
            TipType.EMOTIONAL_MANIPULATION: "emotional manipulation and appeal to feelings",
            TipType.URGENCY: "false urgency and pressure tactics",
            TipType.CONSPIRACY: "conspiracy language and conspiracy theories",
            TipType.GENERAL: "general media literacy and fact-checking"
        }
        
        description = manipulation_descriptions.get(manipulation_type, "media manipulation")
        
        prompt = f"""Create an engaging prebunking tip about {description}.

The tip should help users recognize and avoid this type of manipulation.

"""
        
        if context:
            prompt += f"CONTEXT: {context}\n\n"
        
        prompt += """Make the tip practical, engaging, and easy to remember.
Include real-world examples and actionable advice.

Respond in JSON format as specified in the system prompt."""
        
        return prompt
    
    def _parse_tip_response(self, response_text: str, manipulation_type: TipType) -> PrebunkTip:
        """Parse LLM response into PrebunkTip."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return PrebunkTip(
                    tip_type=manipulation_type,
                    title=data.get('title', f'Tip about {manipulation_type.value}'),
                    description=data.get('description', ''),
                    examples=data.get('examples', []),
                    red_flags=data.get('red_flags', []),
                    how_to_spot=data.get('how_to_spot', []),
                    what_to_do=data.get('what_to_do', []),
                    difficulty=data.get('difficulty', 'medium'),
                    estimated_read_time=data.get('estimated_read_time', 30)
                )
            else:
                return self._generate_template_tip(manipulation_type)
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing tip response: {e}")
            return self._generate_template_tip(manipulation_type)
    
    def _generate_template_tip(self, manipulation_type: TipType) -> PrebunkTip:
        """Generate a template tip when LLM is not available."""
        
        templates = {
            TipType.CLICKBAIT: PrebunkTip(
                tip_type=TipType.CLICKBAIT,
                title="Don't Fall for Clickbait!",
                description="Clickbait uses sensational language to grab your attention, often promising shocking revelations that don't deliver.",
                examples=[
                    "You won't believe what happened next!",
                    "Shocking revelation about...",
                    "This will blow your mind!"
                ],
                red_flags=[
                    "Overly dramatic language",
                    "Promises of shocking revelations",
                    "Exclamation marks everywhere"
                ],
                how_to_spot=[
                    "Look for factual headlines instead",
                    "Check if content delivers on the promise",
                    "Read beyond the headline"
                ],
                what_to_do=[
                    "Seek out balanced reporting",
                    "Check multiple sources",
                    "Look for evidence-based content"
                ],
                difficulty="easy",
                estimated_read_time=20
            ),
            
            TipType.FEAR_APPEAL: PrebunkTip(
                tip_type=TipType.FEAR_APPEAL,
                title="Fear Appeal Alert!",
                description="Fear appeal uses scary language to manipulate your emotions and make you act without thinking.",
                examples=[
                    "This dangerous chemical is in your food!",
                    "Deadly virus spreading rapidly",
                    "Your children are at risk!"
                ],
                red_flags=[
                    "Excessive use of scary words",
                    "Claims of immediate danger",
                    "Emotional manipulation"
                ],
                how_to_spot=[
                    "Look for evidence, not just fear",
                    "Check if the threat is real",
                    "Seek expert opinions"
                ],
                what_to_do=[
                    "Verify claims with credible sources",
                    "Look for balanced perspectives",
                    "Don't let fear cloud your judgment"
                ],
                difficulty="medium",
                estimated_read_time=25
            ),
            
            TipType.CHERRY_PICKING: PrebunkTip(
                tip_type=TipType.CHERRY_PICKING,
                title="Cherry-Picking Detection",
                description="Cherry-picking selectively presents evidence that supports a claim while ignoring contradictory evidence.",
                examples=[
                    "This study proves vaccines are dangerous",
                    "This data shows climate change is fake",
                    "Only this evidence matters"
                ],
                red_flags=[
                    "Claims based on single studies",
                    "Ignoring contradictory evidence",
                    "Selective use of data"
                ],
                how_to_spot=[
                    "Look for the full context",
                    "Check if other studies disagree",
                    "Seek comprehensive analysis"
                ],
                what_to_do=[
                    "Find all available evidence",
                    "Check multiple sources",
                    "Look for consensus among experts"
                ],
                difficulty="medium",
                estimated_read_time=30
            )
        }
        
        return templates.get(manipulation_type, PrebunkTip(
            tip_type=manipulation_type,
            title=f"Understanding {manipulation_type.value.replace('_', ' ').title()}",
            description="Learn to recognize and avoid this manipulation technique.",
            examples=[],
            red_flags=[],
            how_to_spot=[],
            what_to_do=[],
            difficulty="medium",
            estimated_read_time=25
        ))
    
    def generate_educational_card(self, tip: PrebunkTip) -> EducationalCard:
        """
        Generate an educational card from a prebunking tip.
        
        Args:
            tip: PrebunkTip to convert to card
            
        Returns:
            EducationalCard for UI display
        """
        
        # Create engaging content
        content_parts = [
            f"<h3>{tip.title}</h3>",
            f"<p>{tip.description}</p>"
        ]
        
        if tip.examples:
            content_parts.append("<h4>Examples:</h4>")
            content_parts.append("<ul>")
            for example in tip.examples:
                content_parts.append(f"<li>{example}</li>")
            content_parts.append("</ul>")
        
        if tip.red_flags:
            content_parts.append("<h4>Red Flags:</h4>")
            content_parts.append("<ul>")
            for flag in tip.red_flags:
                content_parts.append(f"<li>🚩 {flag}</li>")
            content_parts.append("</ul>")
        
        if tip.how_to_spot:
            content_parts.append("<h4>How to Spot:</h4>")
            content_parts.append("<ul>")
            for method in tip.how_to_spot:
                content_parts.append(f"<li>🔍 {method}</li>")
            content_parts.append("</ul>")
        
        if tip.what_to_do:
            content_parts.append("<h4>What to Do:</h4>")
            content_parts.append("<ul>")
            for action in tip.what_to_do:
                content_parts.append(f"<li>✅ {action}</li>")
            content_parts.append("</ul>")
        
        content = "\n".join(content_parts)
        
        # Determine visual elements based on tip type
        visual_elements = self._get_visual_elements(tip.tip_type)
        
        # Add interactive elements
        interactive_elements = [
            "quiz_button",
            "examples_toggle",
            "practice_scenarios"
        ]
        
        return EducationalCard(
            card_id=f"tip_{tip.tip_type.value}",
            title=tip.title,
            content=content,
            visual_elements=visual_elements,
            interactive_elements=interactive_elements,
            difficulty=tip.difficulty,
            estimated_time=tip.estimated_read_time
        )
    
    def _get_visual_elements(self, tip_type: TipType) -> List[str]:
        """Get appropriate visual elements for tip type."""
        
        visual_map = {
            TipType.CLICKBAIT: ["warning_icon", "headline_examples", "before_after_comparison"],
            TipType.FEAR_APPEAL: ["fear_meter", "calm_icon", "evidence_vs_fear"],
            TipType.CHERRY_PICKING: ["data_visualization", "full_picture_icon", "evidence_scale"],
            TipType.FALSE_AUTHORITY: ["authority_checklist", "credential_icon", "expert_verification"],
            TipType.MISSING_CONTEXT: ["context_puzzle", "full_story_icon", "timeline_view"],
            TipType.EMOTIONAL_MANIPULATION: ["emotion_meter", "fact_icon", "logic_vs_emotion"],
            TipType.URGENCY: ["clock_icon", "pressure_meter", "time_check"],
            TipType.CONSPIRACY: ["conspiracy_scale", "evidence_icon", "fact_check_list"],
            TipType.GENERAL: ["media_literacy_icon", "fact_check_tools", "verification_steps"]
        }
        
        return visual_map.get(tip_type, ["general_icon", "info_panel"])
    
    def generate_quiz_questions(self, tip: PrebunkTip) -> List[Dict[str, Any]]:
        """
        Generate quiz questions for a prebunking tip.
        
        Args:
            tip: PrebunkTip to generate questions for
            
        Returns:
            List of quiz questions
        """
        
        # Template questions based on tip type
        question_templates = {
            TipType.CLICKBAIT: [
                {
                    "question": "Which headline is most likely clickbait?",
                    "options": [
                        "New Study Shows Potential Health Benefits",
                        "You Won't Believe What This Doctor Discovered!",
                        "Research Findings on Diet and Exercise",
                        "Scientific Analysis of Recent Data"
                    ],
                    "correct": 1,
                    "explanation": "The second option uses sensational language typical of clickbait."
                }
            ],
            
            TipType.FEAR_APPEAL: [
                {
                    "question": "What's a red flag for fear appeal?",
                    "options": [
                        "Use of statistics",
                        "Expert opinions",
                        "Excessive scary language",
                        "Balanced reporting"
                    ],
                    "correct": 2,
                    "explanation": "Excessive scary language is a key indicator of fear appeal."
                }
            ],
            
            TipType.CHERRY_PICKING: [
                {
                    "question": "How can you spot cherry-picking?",
                    "options": [
                        "Look for single studies only",
                        "Ignore contradictory evidence",
                        "Check if other studies disagree",
                        "Focus on one source"
                    ],
                    "correct": 2,
                    "explanation": "Checking if other studies disagree helps identify cherry-picking."
                }
            ]
        }
        
        return question_templates.get(tip.tip_type, [
            {
                "question": "What's the best way to verify information?",
                "options": [
                    "Trust the first source you find",
                    "Check multiple credible sources",
                    "Believe what feels right",
                    "Ignore conflicting information"
                ],
                "correct": 1,
                "explanation": "Checking multiple credible sources is the best way to verify information."
            }
        ])
    
    def generate_practice_scenarios(self, tip: PrebunkTip) -> List[Dict[str, Any]]:
        """
        Generate practice scenarios for a prebunking tip.
        
        Args:
            tip: PrebunkTip to generate scenarios for
            
        Returns:
            List of practice scenarios
        """
        
        scenarios = []
        
        # Add scenarios based on examples
        for i, example in enumerate(tip.examples[:3]):  # Limit to 3 scenarios
            scenario = {
                "scenario_id": f"{tip.tip_type.value}_scenario_{i+1}",
                "title": f"Practice Scenario {i+1}",
                "content": example,
                "question": f"Is this an example of {tip.tip_type.value.replace('_', ' ')}?",
                "correct_answer": True,
                "explanation": f"This is an example of {tip.tip_type.value.replace('_', ' ')} because it uses {tip.description.lower()}",
                "difficulty": tip.difficulty
            }
            scenarios.append(scenario)
        
        return scenarios
