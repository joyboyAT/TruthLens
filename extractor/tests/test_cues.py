import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extractor.cues import generate_cue_badges


def test_generate_cue_badges():
	badges = generate_cue_badges(["Clickbait", "Conspiracy"]) 
	assert badges == ["📢 Clickbait", "🎭 Conspiracy"]
