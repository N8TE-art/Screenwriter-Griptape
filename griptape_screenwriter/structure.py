import json
import os
import re
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from griptape.structures import Agent

# PromptStack import is version‑sensitive: utils in 0.23.x, structures in >=0.24
try:
    from griptape.structures import PromptStack  # 0.24+
except ImportError:  # fallback for 0.23.x
    from griptape.utils import PromptStack

from griptape.drivers import OpenAiChatPromptDriver

# -----------------------------------------------------------------------------
# Environment sanity check
# -----------------------------------------------------------------------------
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY environment variable"

# -----------------------------------------------------------------------------
# Shared Pydantic data models
# -----------------------------------------------------------------------------
class Scene(BaseModel):
    act: int
    number: int
    description: str
    conflict: str
    value_change: str

class Outline(BaseModel):
    title: str
    theme: str
    protagonist_desire: str
    protagonist_need: str
    scenes: List[Scene]

class Character(BaseModel):
    name: str
    role: str
    backstory: str
    desire: str
    need: str
    arc: Optional[str] = "Unknown"

class StoryData(BaseModel):
    premise: Optional[str] = None
    outline: Optional[Outline] = None
    characters: List[Character] = []
    analysis_notes: List[str] = []
    screenplay_scenes: List[dict] = []

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
JSON_REGEX = re.compile(r"\{[\s\S]*}?")

def clean_json(raw: str) -> str:
    """Extract first JSON object from a raw LLM response and strip markdown fences."""
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```")
    match = JSON_REGEX.search(raw)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return match.group(0)

def new_driver(model: str = None, temperature: float = 0.3) -> OpenAiChatPromptDriver:
    """Create a fresh driver instance to avoid shared retry state in async contexts."""
    return OpenAiChatPromptDriver(model=model or os.getenv("OPENAI_MODEL", "gpt-4"), temperature=temperature)

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------
class PlotArchitectAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        prompt = f"""
You are a Plot Architect AI. Develop a screenplay outline using Robert McKee's Three‑Act structure.
Premise: "{input_data.premise}"
Follow the structure → Act I (Setup), Act II (Conflict), Act III (Resolution).
Each scene must include: act, number, description, conflict, value_change.
Also include: title, theme, protagonist_desire, protagonist_need.
Respond ONLY with a JSON object.
"""
        response = new_driver().run(PromptStack().add_user_message(prompt))
        try:
            parsed = Outline.parse_raw(response)
        except ValidationError:
            parsed = Outline.parse_raw(clean_json(response))
        return StoryData(premise=input_data.premise, outline=parsed)

class CharacterDesignerAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        assert input_data.outline, "PlotArchitectAgent did not return an outline"
        prompt = f"""
You are a Character Designer AI.
Title: {input_data.outline.title}
Theme: {input_data.outline.theme}
Protagonist's Desire: {input_data.outline.protagonist_desire}
Protagonist's Need: {input_data.outline.protagonist_need}
Design 3‑5 characters (name, role, backstory, desire, need, arc) that reflect or challenge the theme.
Return {{ "characters": [ ... ] }}
"""
        response = new_driver().run(PromptStack().add_user_message(prompt))
        characters_json = json.loads(clean_json(response)).get("characters", [])
        if not characters_json:
            raise ValueError("CharacterDesignerAgent: no 'characters' key in response")
        characters = [Character.parse_obj(c) for c in characters_json]
        return StoryData(premise=input_data.premise, outline=input_data.outline, characters=characters)

class ThematicAnalystAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        assert input_data.characters, "CharacterDesignerAgent did not return characters"
        prompt = f"""
You are a Thematic Analyst AI. Review the outline and characters for coherence.
Respond with {{ "issues_found": bool, "notes": [ ... ] }}
Outline: {input_data.outline.json()}
Characters: {json.dumps([c.dict() for c in input_data.characters])}
"""
        response = new_driver(temperature=0).run(PromptStack().add_user_message(prompt))
        analysis = json.loads(clean_json(response))
        return StoryData(
            premise=input_data.premise,
            outline=input_data.outline,
            characters=input_data.characters,
            analysis_notes=analysis.get("notes", [])
        )

class SceneShaperAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        assert input_data.analysis_notes is not None, "ThematicAnalystAgent missing notes"
        prompt = f"""
You are a Scene Shaper AI. Use the outline, characters, and analyst notes to write screenplay scenes.
Return {{ "scenes": [ {{"number": int, "content": "..."}}, ... ] }}
Outline: {input_data.outline.json()}
Characters: {json.dumps([c.dict() for c in input_data.characters])}
Notes: {json.dumps(input_data.analysis_notes)}
"""
        response = new_driver(temperature=0.7).run(PromptStack().add_user_message(prompt))
        scenes = json.loads(clean_json(response)).get("scenes", [])
        return StoryData(
            premise=input_data.premise,
            outline=input_data.outline,
            characters=input_data.characters,
            analysis_notes=input_data.analysis_notes,
            screenplay_scenes=scenes
        )

# -----------------------------------------------------------------------------
# Pipeline orchestrator
# -----------------------------------------------------------------------------

def run_story_pipeline(premise: str) -> StoryData:
    data = StoryData(premise=premise)
    data = PlotArchitectAgent().run(data)
    data = CharacterDesignerAgent().run(data)
    data = ThematicAnalystAgent().run(data)
    data = SceneShaperAgent().run(data)
    return data

if __name__ == "__main__":
    premise = "A girl discovers her memories have been encoded into a planetary AI network."
    story = run_story_pipeline(premise)
    print("\nTITLE:", story.outline.title)
    print("\nCHARACTERS:", [c.dict() for c in story.characters])
    print("\nNOTES:", story.analysis_notes)
    if story.screenplay_scenes:
        print("\nSCENE 1 PREVIEW:\n", story.screenplay_scenes[0]["content"][:400])
    else:
        print("\nNo scenes generated.")
