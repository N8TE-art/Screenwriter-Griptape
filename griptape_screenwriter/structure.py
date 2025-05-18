import json
import os
import re
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from griptape.structures import Agent
from griptape.common.prompt_stack import PromptStack  # stable 0.23.x location
from griptape.schemas import UserMessage             # official user-role message
from griptape.drivers import OpenAiChatPromptDriver

# -----------------------------------------------------------------------------
# Environment check
# -----------------------------------------------------------------------------
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY environment variable not set"

# -----------------------------------------------------------------------------
# Shared Pydantic models
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
# Utilities
# -----------------------------------------------------------------------------
JSON_RE = re.compile(r"\{[\s\S]*?}\s*$")

def clean_json(raw: str) -> str:
    """Strip markdown fences and extract first JSON object."""
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```")
    match = JSON_RE.search(raw)
    if not match:
        raise ValueError("LLM response contained no valid JSON object")
    return match.group(0)

def new_driver(model: str | None = None, temperature: float = 0.3) -> OpenAiChatPromptDriver:
    return OpenAiChatPromptDriver(model=model or os.getenv("OPENAI_MODEL", "gpt-4"), temperature=temperature)

# -----------------------------------------------------------------------------
# PromptStack helper guaranteed for Griptape 0.23.x
# -----------------------------------------------------------------------------

def stack_with_message(prompt: str) -> PromptStack:
    stack = PromptStack()
    stack.add_message(UserMessage(content=prompt))
    return stack

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------
class PlotArchitectAgent(Agent):
    def run(self, data: StoryData) -> StoryData:
        prompt = (
            "You are a Plot Architect AI. Develop a screenplay outline using Robert McKee's Three-Act structure.\n"
            f"Premise: \"{data.premise}\"\n"
            "Each scene must include: act, number, description, conflict, value_change.\n"
            "Also include: title, theme, protagonist_desire, protagonist_need.\n"
            "Respond ONLY with a JSON object."
        )
        response = new_driver().run(stack_with_message(prompt))
        try:
            outline = Outline.parse_raw(response)
        except ValidationError:
            outline = Outline.parse_raw(clean_json(response))
        return StoryData(premise=data.premise, outline=outline)

class CharacterDesignerAgent(Agent):
    def run(self, data: StoryData) -> StoryData:
        assert data.outline, "PlotArchitectAgent did not produce an outline"
        prompt = (
            "You are a Character Designer AI.\n"
            f"Title: {data.outline.title}\n"
            f"Theme: {data.outline.theme}\n"
            f"Protagonist's Desire: {data.outline.protagonist_desire}\n"
            f"Protagonist's Need: {data.outline.protagonist_need}\n"
            "Design 3-5 characters (name, role, backstory, desire, need, arc).\n"
            "Return { \"characters\": [ ... ] }."
        )
        response = new_driver().run(stack_with_message(prompt))
        chars_json = json.loads(clean_json(response)).get("characters", [])
        if not chars_json:
            raise ValueError("CharacterDesignerAgent response missing 'characters'")
        characters = [Character.parse_obj(c) for c in chars_json]
        return StoryData(premise=data.premise, outline=data.outline, characters=characters)

class ThematicAnalystAgent(Agent):
    def run(self, data: StoryData) -> StoryData:
        assert data.characters, "CharacterDesignerAgent produced no characters"
        prompt = (
            "You are a Thematic Analyst AI. Review the outline and characters for coherence.\n"
            "Return { \"issues_found\": bool, \"notes\": [...] }.\n"
            f"Outline: {data.outline.json()}\n"
            f"Characters: {json.dumps([c.dict() for c in data.characters])}"
        )
        response = new_driver(temperature=0).run(stack_with_message(prompt))
        analysis = json.loads(clean_json(response))
        return StoryData(premise=data.premise, outline=data.outline, characters=data.characters, analysis_notes=analysis.get("notes", []))

class SceneShaperAgent(Agent):
    def run(self, data: StoryData) -> StoryData:
        prompt = (
            "You are a Scene Shaper AI. Write screenplay scenes based on the outline, characters, and analyst notes.\n"
            "Return { \"scenes\": [ {\"number\": int, \"content\": \"...\"}, ... ] }.\n"
            f"Outline: {data.outline.json()}\n"
            f"Characters: {json.dumps([c.dict() for c in data.characters])}\n"
            f"Notes: {json.dumps(data.analysis_notes)}"
        )
        response = new_driver(temperature=0.7).run(stack_with_message(prompt))
        scenes = json.loads(clean_json(response)).get("scenes", [])
        return StoryData(premise=data.premise, outline=data.outline, characters=data.characters, analysis_notes=data.analysis_notes, screenplay_scenes=scenes)

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
    final_story = run_story_pipeline(premise)
    print("TITLE:", final_story.outline.title)
    print("CHARACTERS:", [c.dict() for c in final_story.characters])
    print("NOTES:", final_story.analysis_notes)
    if final_story.screenplay_scenes:
        print("SCENE 1 PREVIEW:\n", final_story.screenplay_scenes[0]["content"][:400])
    else:
        print("No scenes generated.")
