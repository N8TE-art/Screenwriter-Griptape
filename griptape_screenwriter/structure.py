import json
import os
import re
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from griptape.structures import Agent

# -----------------------------------------------------------------------------
# PromptStack compatibility across Griptape versions (0.23.x stable preferred)
# -----------------------------------------------------------------------------
try:
    from griptape.common.prompt_stack import PromptStack  # 0.23.x
except ImportError:
    try:
        from griptape.structures import PromptStack  # ≥0.24 (dev/nightly)
    except ImportError:
        from griptape.utils import PromptStack  # legacy fallback

from griptape.drivers import OpenAiChatPromptDriver

# -----------------------------------------------------------------------------
# Environment sanity check
# -----------------------------------------------------------------------------
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY environment variable"

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
# Helper utilities
# -----------------------------------------------------------------------------
JSON_REGEX = re.compile(r"\{[\s\S]*}?")

def clean_json(raw: str) -> str:
    raw = raw.strip().lstrip("```json").lstrip("```")
    match = JSON_REGEX.search(raw)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return match.group(0)

def new_driver(model: str = None, temperature: float = 0.3) -> OpenAiChatPromptDriver:
    return OpenAiChatPromptDriver(model=model or os.getenv("OPENAI_MODEL", "gpt-4"), temperature=temperature)


def stack_with_message(prompt: str) -> PromptStack:
    """Return a PromptStack containing a single user message in the most
    version‑agnostic way possible (0.22 → 0.24)."""
    # Preferred shortcut if factory exists
    if hasattr(PromptStack, "from_artifact"):
        return PromptStack.from_artifact(prompt)

    stack = PromptStack()
    # Newer helper (0.23+)
    if hasattr(stack, "add_user_message"):
        stack.add_user_message(prompt)
    # Older generic helper (pre‑0.23 nightly)
    elif hasattr(stack, "add_message"):
        stack.add_message(prompt, "user")
    else:
        # Ultimate fallback: try .inputs list used by drivers
        if hasattr(stack, "messages"):
            stack.messages.append({"role": "user", "content": prompt})
        elif hasattr(stack, "inputs"):
            stack.inputs.append(prompt)  # very old prototype
        else:
            raise AttributeError("PromptStack has no method to add a user message in this Griptape version")
    return stack

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------
class PlotArchitectAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        prompt = (
            "You are a Plot Architect AI. Develop a screenplay outline using Robert McKee's Three-Act structure.\n"
            f"Premise: \"{input_data.premise}\"\n"
            "Each scene must include: act, number, description, conflict, value_change.\n"
            "Also include: title, theme, protagonist_desire, protagonist_need.\n"
            "Respond ONLY with a JSON object."
        )
        response = new_driver().run(stack_with_message(prompt))
        try:
            outline = Outline.parse_raw(response)
        except ValidationError:
            outline = Outline.parse_raw(clean_json(response))
        return StoryData(premise=input_data.premise, outline=outline)

class CharacterDesignerAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        assert input_data.outline, "PlotArchitectAgent failed to produce outline"
        prompt = (
            "You are a Character Designer AI.\n"
            f"Title: {input_data.outline.title}\n"
            f"Theme: {input_data.outline.theme}\n"
            f"Protagonist's Desire: {input_data.outline.protagonist_desire}\n"
            f"Protagonist's Need: {input_data.outline.protagonist_need}\n"
            "Design 3‑5 characters (name, role, backstory, desire, need, arc) that reflect or challenge the theme.\n"
            "Return { \"characters\": [ ... ] }"
        )
        response = new_driver().run(stack_with_message(prompt))
        chars_json = json.loads(clean_json(response)).get("characters", [])
        if not chars_json:
            raise ValueError("CharacterDesignerAgent: missing 'characters' in response")
        characters = [Character.parse_obj(c) for c in chars_json]
        return StoryData(premise=input_data.premise, outline=input_data.outline, characters=characters)

class ThematicAnalystAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        assert input_data.characters, "CharacterDesignerAgent returned empty characters"
        prompt = (
            "You are a Thematic Analyst AI. Review the outline and characters for coherence.\n"
            "Return { \"issues_found\": bool, \"notes\": [...] }.\n"
            f"Outline: {input_data.outline.json()}\n"
            f"Characters: {json.dumps([c.dict() for c in input_data.characters])}"
        )
        response = new_driver(temperature=0).run(stack_with_message(prompt))
        analysis = json.loads(clean_json(response))
        return StoryData(premise=input_data.premise, outline=input_data.outline, characters=input_data.characters, analysis_notes=analysis.get("notes", []))

class SceneShaperAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        prompt = (
            "You are a Scene Shaper AI. Use the outline, characters, and analyst notes to write screenplay scenes.\n"
            "Return { \"scenes\": [ {\"number\": int, \"content\": \"...\"}, ... ] }.\n"
            f"Outline: {input_data.outline.json()}\n"
            f"Characters: {json.dumps([c.dict() for c in input_data.characters])}\n"
            f"Notes: {json.dumps(input_data.analysis_notes)}"
        )
        response = new_driver(temperature=0.7).run(stack_with_message(prompt))
        scenes = json.loads(clean_json(response)).get("scenes", [])
        return StoryData(premise=input_data.premise, outline=input_data.outline, characters=input_data.characters, analysis_notes=input_data.analysis_notes, screenplay_scenes=scenes)

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
