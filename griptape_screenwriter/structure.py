import json
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from griptape.structures import Agent
from griptape.utils import PromptStack
from griptape.drivers import OpenAiChatPromptDriver

# Shared schema using Pydantic
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
    arc: str

class StoryData(BaseModel):
    premise: Optional[str] = None
    outline: Optional[Outline] = None
    characters: List[Character] = []
    analysis_notes: List[str] = []
    screenplay_scenes: List[dict] = []

# Default driver
DEFAULT_DRIVER = OpenAiChatPromptDriver(model="gpt-4", temperature=0.3)

class PlotArchitectAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        prompt = f"""
You are a Plot Architect AI. Develop a screenplay outline using Robert McKee's Three-Act structure.

Premise: "{input_data.premise}"

Follow the structure:
- Act I (Setup): Inciting Incident
- Act II (Conflict): Rising conflict, midpoint, major crisis
- Act III (Resolution): Climax and transformation

Each scene must include:
- act (1|2|3)
- number
- description
- conflict
- value_change

Also include:
- title
- theme
- protagonist_desire
- protagonist_need

Respond ONLY with a valid JSON object.
"""
        response = DEFAULT_DRIVER.run(PromptStack().add_user_message(prompt))
        try:
            parsed = Outline.parse_raw(response)
        except ValidationError:
            parsed = Outline.parse_obj(json.loads(response.strip('`')))
        return StoryData(premise=input_data.premise, outline=parsed)

class CharacterDesignerAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        prompt = f"""
You are a Character Designer AI.

Title: {input_data.outline.title}
Theme: {input_data.outline.theme}
Protagonist's Desire: {input_data.outline.protagonist_desire}
Protagonist's Need: {input_data.outline.protagonist_need}

Design 3–5 characters who reflect or challenge the theme.
For each, include: name, role, backstory, desire, need, arc.
Respond ONLY with: {{ "characters": [ {{...}}, ... ] }}
"""
        response = DEFAULT_DRIVER.run(PromptStack().add_user_message(prompt))
        parsed = json.loads(response)
        characters = [Character.parse_obj(c) for c in parsed["characters"]]
        return StoryData(
            premise=input_data.premise,
            outline=input_data.outline,
            characters=characters
        )

class ThematicAnalystAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        prompt = f"""
You are a Thematic Analyst AI reviewing a screenplay.

Title: {input_data.outline.title}
Theme: {input_data.outline.theme}

Characters:
{json.dumps([c.dict() for c in input_data.characters], indent=2)}

Scenes:
{json.dumps([s.dict() for s in input_data.outline.scenes], indent=2)}

Evaluate each scene:
- Is conflict clear?
- Is there a value change?
- Does it reflect or oppose the theme?
Check if protagonist is challenged to confront their Need.

Respond with:
{{ "issues_found": true/false, "notes": ["..."] }}
"""
        response = DEFAULT_DRIVER.run(PromptStack().add_user_message(prompt))
        parsed = json.loads(response)
        return StoryData(
            premise=input_data.premise,
            outline=input_data.outline,
            characters=input_data.characters,
            analysis_notes=parsed.get("notes", [])
        )

class SceneShaperAgent(Agent):
    def run(self, input_data: StoryData) -> StoryData:
        prompt = f"""
You are a Scene Shaper AI. Write screenplay scenes.

Title: {input_data.outline.title}
Theme: {input_data.outline.theme}

Characters:
{json.dumps([c.dict() for c in input_data.characters], indent=2)}

Scenes:
{json.dumps([s.dict() for s in input_data.outline.scenes], indent=2)}

Analyst Notes:
{json.dumps(input_data.analysis_notes)}

Write formatted screenplay scenes. Respond with:
{{ "scenes": [ {{"number": int, "content": "Scene text..."}}, ... ] }}
"""
        response = DEFAULT_DRIVER.run(PromptStack().add_user_message(prompt))
        parsed = json.loads(response)
        return StoryData(
            premise=input_data.premise,
            outline=input_data.outline,
            characters=input_data.characters,
            analysis_notes=input_data.analysis_notes,
            screenplay_scenes=parsed.get("scenes", [])
        )

def run_story_pipeline(premise: str) -> StoryData:
    data = StoryData(premise=premise)
    data = PlotArchitectAgent().run(data)
    data = CharacterDesignerAgent().run(data)
    data = ThematicAnalystAgent().run(data)
    data = SceneShaperAgent().run(data)
    return data

if __name__ == "__main__":
    sample_premise = "A girl discovers her memories have been encoded into a planetary AI network."
    result = run_story_pipeline(sample_premise)
    print("\nTITLE:\n", result.outline.title)
    print("\nCHARACTERS:\n", [c.dict() for c in result.characters])
    print("\nNOTES:\n", result.analysis_notes)
    print("\nSCREENPLAY EXCERPT:\n", result.screenplay_scenes[0]["content"][:500] if result.screenplay_scenes else "No scenes generated.")
