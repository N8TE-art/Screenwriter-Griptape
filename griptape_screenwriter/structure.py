import json
import sys
import os
from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from pydantic import BaseModel, Field
from typing import List

# --- SCHEMAS ---
class PlotInput(BaseModel):
    premise: str = Field(..., description="Story premise")

class CharInput(BaseModel):
    premise: str = Field(..., description="Story premise")
    outline: str = Field(..., description="Plot outline")

class ThemeInput(BaseModel):
    outline: str = Field(..., description="Plot outline")
    characters: str = Field(..., description="Character list")

class ScriptInput(BaseModel):
    premise: str = Field(..., description="Story premise")
    outline: str = Field(..., description="Plot outline")
    characters: str = Field(..., description="Character list")
    transitions: str = Field(..., description="Value transitions")

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
        "Respond ONLY with raw JSON. Do not include any explanations.\n"
        "You are a Plot Architect writing a screenplay outline based on the premise: '{{ input.premise }}'.\n"
        "Provide a one-sentence LOGLINE and a STORY OUTLINE as a list of scenes.\n"
        "Return an object with keys: 'logline' (string), 'outline' (array of scene descriptions)."
    )
    return PromptTask(
        id="plot_architect",
        prompt_template=prompt,
        input_schema=PlotInput,
        output_schema=PlotOutline,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_character_designer():
    prompt = (
        "Respond ONLY with raw JSON. Do not include any explanations.\n"
        "You are a Character Designer. Given the premise '{{ input.premise }}' and outline:\n{{ input.outline }}\n"
        "Generate character profiles as JSON with fields: name, role, description, and arc."
    )
    return PromptTask(
        id="character_designer",
        prompt_template=prompt,
        input_schema=CharInput,
        output_schema=CharacterList,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_thematic_analyst():
    prompt = (
        "Respond ONLY with raw JSON. Do not include any explanations.\n"
        "You are a Thematic Analyst. Given the outline:\n{{ input.outline }}\nand characters:\n{{ input.characters }}\n"
        "List each scene's value change as JSON: scene, from_value, to_value."
    )
    return PromptTask(
        id="thematic_analyst",
        prompt_template=prompt,
        input_schema=ThemeInput,
        output_schema=ValueTransitions,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    )

def build_scene_shaper():
    prompt = (
        "Respond ONLY with raw JSON. Do not include any explanations.\n"
        "You are a Screenwriter. Write a full screenplay using:\n"
        "Premise: {{ input.premise }}\n"
        "Outline: {{ input.outline }}\n"
        "Characters: {{ input.characters }}\n"
        "Value Transitions: {{ input.transitions }}\n"
        "Return the complete script in one text block."
    )
    return PromptTask(
        id="scene_shaper",
        prompt_template=prompt,
        input_schema=ScriptInput,
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

    # Debug function to inspect task outputs
    def debug(task_id, label):
        task = workflow.find_task_by_id(task_id)
        if task and task.output:
            workflow.events.append({
                "type": "debug",
                "payload": f"{label} Output:\n{task.output.value}"
            })
        else:
            workflow.events.append({
                "type": "debug",
                "payload": f"{label} Output: None or Invalid"
            })

    workflow.run()

    debug(plot_task.id, "PLOT")
    debug(char_task.id, "CHARACTERS")
    debug(theme_task.id, "
