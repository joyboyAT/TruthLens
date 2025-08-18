"""
Tests for prebunk generator module.
"""

import pytest
from src.manipulation.prebunk_generator import (
    PrebunkGenerator,
    TipType,
    PrebunkTip,
    EducationalCard
)


class TestPrebunkGenerator:
    """Test cases for PrebunkGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PrebunkGenerator(use_llm=False)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator is not None
        assert self.generator.use_llm is False
    
    def test_generate_template_tip_clickbait(self):
        """Test generating clickbait tip template."""
        tip = self.generator._generate_template_tip(TipType.CLICKBAIT)
        
        assert isinstance(tip, PrebunkTip)
        assert tip.tip_type == TipType.CLICKBAIT
        assert tip.title == "Don't Fall for Clickbait!"
        assert "sensational language" in tip.description.lower()
        assert len(tip.examples) > 0
        assert len(tip.red_flags) > 0
        assert len(tip.how_to_spot) > 0
        assert len(tip.what_to_do) > 0
        assert tip.difficulty == "easy"
        assert tip.estimated_read_time > 0
    
    def test_generate_template_tip_fear_appeal(self):
        """Test generating fear appeal tip template."""
        tip = self.generator._generate_template_tip(TipType.FEAR_APPEAL)
        
        assert isinstance(tip, PrebunkTip)
        assert tip.tip_type == TipType.FEAR_APPEAL
        assert tip.title == "Fear Appeal Alert!"
        assert "scary language" in tip.description.lower()
        assert len(tip.examples) > 0
        assert tip.difficulty == "medium"
    
    def test_generate_template_tip_cherry_picking(self):
        """Test generating cherry picking tip template."""
        tip = self.generator._generate_template_tip(TipType.CHERRY_PICKING)
        
        assert isinstance(tip, PrebunkTip)
        assert tip.tip_type == TipType.CHERRY_PICKING
        assert tip.title == "Cherry-Picking Detection"
        assert "selectively presents" in tip.description.lower()
        assert len(tip.examples) > 0
    
    def test_generate_template_tip_unknown_type(self):
        """Test generating tip for unknown manipulation type."""
        tip = self.generator._generate_template_tip(TipType.GENERAL)
        
        assert isinstance(tip, PrebunkTip)
        assert tip.tip_type == TipType.GENERAL
        assert "Understanding" in tip.title
        assert len(tip.examples) == 0  # Template has no examples
    
    def test_generate_tip_without_llm(self):
        """Test generating tip without LLM (uses templates)."""
        tip = self.generator.generate_tip(TipType.CLICKBAIT)
        
        assert isinstance(tip, PrebunkTip)
        assert tip.tip_type == TipType.CLICKBAIT
        assert tip.title == "Don't Fall for Clickbait!"
    
    def test_generate_educational_card(self):
        """Test generating educational card from tip."""
        tip = self.generator.generate_tip(TipType.CLICKBAIT)
        card = self.generator.generate_educational_card(tip)
        
        assert isinstance(card, EducationalCard)
        assert card.card_id == f"tip_{tip.tip_type.value}"
        assert card.title == tip.title
        assert len(card.content) > 0
        assert len(card.visual_elements) > 0
        assert len(card.interactive_elements) > 0
        assert card.difficulty == tip.difficulty
        assert card.estimated_time == tip.estimated_read_time
    
    def test_get_visual_elements(self):
        """Test getting visual elements for different tip types."""
        visual_elements = self.generator._get_visual_elements(TipType.CLICKBAIT)
        
        assert isinstance(visual_elements, list)
        assert len(visual_elements) > 0
        assert "warning_icon" in visual_elements
        assert "headline_examples" in visual_elements
    
    def test_get_visual_elements_unknown_type(self):
        """Test getting visual elements for unknown tip type."""
        visual_elements = self.generator._get_visual_elements(TipType.GENERAL)

        assert isinstance(visual_elements, list)
        assert len(visual_elements) > 0
        assert "media_literacy_icon" in visual_elements
    
    def test_generate_quiz_questions(self):
        """Test generating quiz questions for tip."""
        tip = self.generator.generate_tip(TipType.CLICKBAIT)
        questions = self.generator.generate_quiz_questions(tip)
        
        assert isinstance(questions, list)
        assert len(questions) > 0
        
        question = questions[0]
        assert "question" in question
        assert "options" in question
        assert "correct" in question
        assert "explanation" in question
        assert len(question["options"]) > 0
        assert isinstance(question["correct"], int)
    
    def test_generate_practice_scenarios(self):
        """Test generating practice scenarios for tip."""
        tip = self.generator.generate_tip(TipType.CLICKBAIT)
        scenarios = self.generator.generate_practice_scenarios(tip)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        
        scenario = scenarios[0]
        assert "scenario_id" in scenario
        assert "title" in scenario
        assert "content" in scenario
        assert "question" in scenario
        assert "correct_answer" in scenario
        assert "explanation" in scenario
        assert "difficulty" in scenario


class TestTipType:
    """Test cases for TipType enum."""
    
    def test_enum_values(self):
        """Test that all tip types are defined."""
        expected_types = {
            'CLICKBAIT',
            'FEAR_APPEAL',
            'CHERRY_PICKING',
            'FALSE_AUTHORITY',
            'MISSING_CONTEXT',
            'EMOTIONAL_MANIPULATION',
            'URGENCY',
            'CONSPIRACY',
            'GENERAL'
        }
        
        actual_types = {t.name for t in TipType}
        assert actual_types == expected_types
    
    def test_enum_string_values(self):
        """Test that enum values are properly formatted strings."""
        for tip_type in TipType:
            assert isinstance(tip_type.value, str)
            # Values should be lowercase and use underscores or be single words
            assert tip_type.value.islower()


class TestPrebunkTip:
    """Test cases for PrebunkTip dataclass."""
    
    def test_tip_creation(self):
        """Test creating a PrebunkTip."""
        tip = PrebunkTip(
            tip_type=TipType.CLICKBAIT,
            title="Test Tip",
            description="Test description",
            examples=["Example 1", "Example 2"],
            red_flags=["Flag 1"],
            how_to_spot=["Method 1"],
            what_to_do=["Action 1"],
            difficulty="easy",
            estimated_read_time=30
        )
        
        assert tip.tip_type == TipType.CLICKBAIT
        assert tip.title == "Test Tip"
        assert tip.description == "Test description"
        assert tip.examples == ["Example 1", "Example 2"]
        assert tip.red_flags == ["Flag 1"]
        assert tip.how_to_spot == ["Method 1"]
        assert tip.what_to_do == ["Action 1"]
        assert tip.difficulty == "easy"
        assert tip.estimated_read_time == 30


class TestEducationalCard:
    """Test cases for EducationalCard dataclass."""
    
    def test_card_creation(self):
        """Test creating an EducationalCard."""
        card = EducationalCard(
            card_id="test_card",
            title="Test Card",
            content="<h3>Test Content</h3>",
            visual_elements=["icon1", "icon2"],
            interactive_elements=["button1"],
            difficulty="medium",
            estimated_time=45
        )
        
        assert card.card_id == "test_card"
        assert card.title == "Test Card"
        assert card.content == "<h3>Test Content</h3>"
        assert card.visual_elements == ["icon1", "icon2"]
        assert card.interactive_elements == ["button1"]
        assert card.difficulty == "medium"
        assert card.estimated_time == 45


class TestContentGeneration:
    """Test cases for content generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PrebunkGenerator(use_llm=False)
    
    def test_educational_card_content_structure(self):
        """Test that educational card content has proper HTML structure."""
        tip = self.generator.generate_tip(TipType.CLICKBAIT)
        card = self.generator.generate_educational_card(tip)
        
        content = card.content
        
        # Should contain HTML elements
        assert "<h3>" in content
        assert "<p>" in content
        
        # Should contain tip information
        assert tip.title in content
        assert tip.description in content
        
        # Should contain lists if examples exist
        if tip.examples:
            assert "<ul>" in content
            assert "<li>" in content
    
    def test_quiz_question_structure(self):
        """Test that quiz questions have proper structure."""
        tip = self.generator.generate_tip(TipType.CLICKBAIT)
        questions = self.generator.generate_quiz_questions(tip)
        
        for question in questions:
            # Check required fields
            assert "question" in question
            assert "options" in question
            assert "correct" in question
            assert "explanation" in question
            
            # Check data types
            assert isinstance(question["question"], str)
            assert isinstance(question["options"], list)
            assert isinstance(question["correct"], int)
            assert isinstance(question["explanation"], str)
            
            # Check data validity
            assert len(question["options"]) > 0
            assert 0 <= question["correct"] < len(question["options"])
    
    def test_practice_scenario_structure(self):
        """Test that practice scenarios have proper structure."""
        tip = self.generator.generate_tip(TipType.CLICKBAIT)
        scenarios = self.generator.generate_practice_scenarios(tip)
        
        for scenario in scenarios:
            # Check required fields
            assert "scenario_id" in scenario
            assert "title" in scenario
            assert "content" in scenario
            assert "question" in scenario
            assert "correct_answer" in scenario
            assert "explanation" in scenario
            assert "difficulty" in scenario
            
            # Check data types
            assert isinstance(scenario["scenario_id"], str)
            assert isinstance(scenario["title"], str)
            assert isinstance(scenario["content"], str)
            assert isinstance(scenario["question"], str)
            assert isinstance(scenario["correct_answer"], bool)
            assert isinstance(scenario["explanation"], str)
            assert isinstance(scenario["difficulty"], str)


if __name__ == "__main__":
    pytest.main([__file__])
