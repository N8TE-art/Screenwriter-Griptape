from pydantic import BaseModel, Field
from typing import List, Optional
from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.memory import TextMemory
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver

# --- UNIFYING SCHEMA ---
class Scene(BaseModel):
    act: int
    number: int
    description: str
    conflict: str
    value_change: str

class CharacterProfile(BaseModel):
    name: str
    role: str
    backstory: str
    desire: str
    need: str
    arc: str

class SceneScript(BaseModel):
    number: int
    content: str

class StoryContext(BaseModel):
    premise: str
    title: Optional[str] = None
    theme: Optional[str] = None
    protagonist_desire: Optional[str] = None
    protagonist_need: Optional[str] = None
    scenes: Optional[List[Scene]] = []
    characters: Optional[List[CharacterProfile]] = []
    notes: Optional[List[str]] = []
    issues_found: Optional[bool] = False
    screenplay: Optional[List[SceneScript]] = []

# --- WORKFLOW DEFINITION ---
def build_workflow():
    memory = TextMemory()

    plot_task = PromptTask(
        id="plot_architect",
        prompt_template="{{ args[0] }}",
        input_schema=type("Premise", (BaseModel,), {"premise": (str, ...)}),
        output_schema=StoryContext,
        memory=memory,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4")
    )

    char_task = PromptTask(
        id="character_designer",
        prompt_template="{{ tasks.plot_architect.output }}",
        input_schema=StoryContext,
        output_schema=StoryContext,
        memory=memory,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4")
    )

    theme_task = PromptTask(
        id="thematic_analyst",
        prompt_template="""
        {
            "outline": {{ tasks.plot_architect.output }},
            "characters": {{ tasks.character_designer.output }}
        }
        """,
        input_schema=StoryContext,
        output_schema=StoryContext,
        memory=memory,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4")
    )

    scene_task = PromptTask(
        id="scene_shaper",
        prompt_template="""
        {
            "premise": {{ args[0] }},
            "outline": {{ tasks.plot_architect.output }},
            "characters": {{ tasks.character_designer.output }},
            "notes": {{ tasks.thematic_analyst.output.notes }}
        }
        """,
        input_schema=StoryContext,
        output_schema=StoryContext,
        memory=memory,
        prompt_driver=OpenAiChatPromptDriver(model="gpt-4")
    )

    workflow = Workflow()
    workflow.add_task(plot_task)
    workflow.add_task(char_task, parent_task_id=plot_task.id)
    workflow.add_task(theme_task, parent_task_id=char_task.id)
    workflow.add_task(scene_task, parent_task_id=theme_task.id)

    workflow.output_task_id = scene_task.id
    return workflow
