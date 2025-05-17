# structure.py
from griptape.structures import Workflow
from griptape.tasks import StructureRunTask
from griptape.drivers import OpenAiChatPromptDriver
from griptape.drivers.structure_run.local import LocalStructureRunDriver
from pydantic import BaseModel, Field
from typing import List
import os

# --- SCHEMAS ---
class SceneOutline(BaseModel):
    number: int
    description: str
    value_change: str

class PlotOutline(BaseModel):
    title: str
    theme: str
    protagonist_desire: str
    protagonist_need: str
    scenes: List[SceneOutline]

class CharacterProfile(BaseModel):
    name: str
    role: str
    backstory: str
    desire: str
    need: str
    arc: str

class CharacterProfiles(BaseModel):
    characters: List[CharacterProfile]

class StoryContext(BaseModel):
    outline: PlotOutline
    characters: CharacterProfiles

class StoryAnalysis(BaseModel):
    issues_found: bool
    notes: List[str]

class SceneScript(BaseModel):
    number: int
    content: str

class Screenplay(BaseModel):
    scenes: List[SceneScript]

# --- AGENTS ---
def build_plot_architect():
    prompt = (
        "You are a Plot Architect AI. Develop a screenplay outline from the given premise."
        " Each scene must involve conflict and end with a value change."
        " Track protagonist's Desire vs Need. Output JSON with title, theme, protagonist_desire, protagonist_need, and scenes."
        "\n\nPremise: {{ input.premise }}"
    )
    return lambda: StructureRunTask(
        id="plot_architect",
        prompt_template=prompt,
        input_schema=type("PlotInput", (BaseModel,), {"premise": (str, ...)}),
        output_schema=PlotOutline,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_character_designer():
    prompt = (
        "You are a Character Designer. Given a story outline, create character profiles."
        " Include name, role, backstory, desire, need, and arc. Output as JSON."
        "\n\nOutline: {{ input.outline }}"
    )
    return lambda: StructureRunTask(
        id="character_designer",
        prompt_template=prompt,
        input_schema=type("CharInput", (BaseModel,), {"outline": (PlotOutline, ...)}),
        output_schema=CharacterProfiles,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_thematic_analyst():
    prompt = (
        "You are a Thematic Analyst AI. Analyze the story for alignment with the theme."
        " Check each scene for value change, character arc consistency, and theme support."
        "\n\nOutline: {{ input.outline }}\nCharacters: {{ input.characters }}"
    )
    return lambda: StructureRunTask(
        id="thematic_analyst",
        prompt_template=prompt,
        input_schema=StoryContext,
        output_schema=StoryAnalysis,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_scene_shaper():
    prompt = (
        "You are a Scene Shaper AI. Write full screenplay scenes from the outline and character profiles."
        " Each scene must reflect the value change described and use proper screenplay formatting."
        "\n\nOutline: {{ input.outline }}\nCharacters: {{ input.characters }}"
    )
    return lambda: StructureRunTask(
        id="scene_shaper",
        prompt_template=prompt,
        input_schema=StoryContext,
        output_schema=Screenplay,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

# --- WORKFLOW ---
def run_workflow():
    workflow = Workflow()

    premise_input = {"premise": "A timid farmer must confront a dragon to save his village."}

    plot_task = workflow.add_task(build_plot_architect()(), input=premise_input)

    char_task = workflow.add_task(build_character_designer()(),
                                  input={"outline": "{{ tasks.plot_architect.output }}"},
                                  parent_task_id=plot_task.id)

    analysis_task = workflow.add_task(build_thematic_analyst()(),
                                      input={"outline": "{{ tasks.plot_architect.output }}",
                                             "characters": "{{ tasks.character_designer.output }}"},
                                      parent_task_id=char_task.id)

    scene_task = workflow.add_task(build_scene_shaper()(),
                                   input={"outline": "{{ tasks.plot_architect.output }}",
                                          "characters": "{{ tasks.character_designer.output }}"},
                                   parent_task_id=analysis_task.id)

    return workflow
