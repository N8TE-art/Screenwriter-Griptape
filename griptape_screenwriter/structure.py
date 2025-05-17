import json
import sys
import os
from griptape.structures import Workflow
from griptape.tasks import StructureRunTask
from griptape.drivers import OpenAiChatPromptDriver
from pydantic import BaseModel, Field
from typing import List

# --- SCHEMAS ---
class PlotOutline(BaseModel):
    logline: str
    outline: List[str]

class CharacterProfile(BaseModel):
    name: str
    role: str
    description: str
    arc: str

class CharacterList(BaseModel):
    characters: List[CharacterProfile]

class ValueTransition(BaseModel):
    scene: int
    from_value: str
    to_value: str

class ValueTransitions(BaseModel):
    transitions: List[ValueTransition]

class Screenplay(BaseModel):
    script: str

# --- AGENTS ---
def build_plot_architect():
    prompt = (
        "You are a Plot Architect writing a screenplay outline based on the premise: '{{ input.premise }}'.\n"
        "Provide a one-sentence LOGLINE and a STORY OUTLINE as a list of scenes.\n"
        "Respond in JSON with keys 'logline' (string) and 'outline' (array of scene descriptions)."
    )
    input_schema = type("PlotInput", (BaseModel,), {"premise": (str, ...)})
    return StructureRunTask(
        id="plot_architect",
        prompt_template=prompt,
        input_schema=input_schema,
        output_schema=PlotOutline,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_character_designer():
    prompt = (
        "You are a Character Designer. Given the premise '{{ input.premise }}' and outline:\n{{ input.outline }}\n"
        "Generate character profiles as JSON with fields: name, role, description, and arc."
    )
    input_schema = type("CharInput", (BaseModel,), {
        "premise": (str, ...),
        "outline": (str, ...)
    })
    return StructureRunTask(
        id="character_designer",
        prompt_template=prompt,
        input_schema=input_schema,
        output_schema=CharacterList,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_thematic_analyst():
    prompt = (
        "You are a Thematic Analyst. Given the outline:\n{{ input.outline }}\nand characters:\n{{ input.characters }}\n"
        "List each scene's value change as JSON: scene, from_value, to_value."
    )
    input_schema = type("ThemeInput", (BaseModel,), {
        "outline": (str, ...),
        "characters": (str, ...)
    })
    return StructureRunTask(
        id="thematic_analyst",
        prompt_template=prompt,
        input_schema=input_schema,
        output_schema=ValueTransitions,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_scene_shaper():
    prompt = (
        "You are a Screenwriter. Write a full screenplay using:\nPremise: {{ input.premise }}\nOutline: {{ input.outline }}\nCharacters: {{ input.characters }}\nValue Transitions: {{ input.transitions }}\n"
        "Return the complete script in one text block."
    )
    input_schema = type("ScriptInput", (BaseModel,), {
        "premise": (str, ...),
        "outline": (str, ...),
        "characters": (str, ...),
        "transitions": (str, ...)
    })
    return StructureRunTask(
        id="scene_shaper",
        prompt_template=prompt,
        input_schema=input_schema,
        output_schema=Screenplay,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

# --- WORKFLOW ---
def run_workflow():
    workflow = Workflow()

    if len(sys.argv) < 2:
        print("Usage: python structure.py \"<premise>\"")
        sys.exit(1)

    premise = sys.argv[1]

    plot_task = workflow.add_task(build_plot_architect(), input={"premise": premise})
    char_task = workflow.add_task(
        build_character_designer(),
        input={"premise": premise, "outline": "{{ tasks.plot_architect.output }}"},
        parent_task_id=plot_task.id
    )
    theme_task = workflow.add_task(
        build_thematic_analyst(),
        input={"outline": "{{ tasks.plot_architect.output }}", "characters": "{{ tasks.character_designer.output }}"},
        parent_task_id=char_task.id
    )
    scene_task = workflow.add_task(
        build_scene_shaper(),
        input={
            "premise": premise,
            "outline": "{{ tasks.plot_architect.output }}",
            "characters": "{{ tasks.character_designer.output }}",
            "transitions": "{{ tasks.thematic_analyst.output }}"
        },
        parent_task_id=theme_task.id
    )

    workflow.output_task_id = scene_task.id
    return workflow
