```python
from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.drivers import OpenAiChatPromptDriver
from pydantic import BaseModel

# 1. Define Pydantic models for structured outputs (schemas)
class Scene(BaseModel):
    description: str           # brief description of scene events
    value_change: str          # the key emotional or narrative shift in the scene

class PlotOutline(BaseModel):
    theme: str
    protagonist_want: str
    protagonist_need: str
    scenes: list[Scene]

class CharacterProfile(BaseModel):
    name: str
    role: str                  # e.g., protagonist, antagonist, mentor, etc.
    backstory: str
    desire: str                # what the character wants
    need: str                  # what the character truly needs to learn
    arc: str                   # summary of the character's arc over the story

class CharacterProfiles(BaseModel):
    characters: list[CharacterProfile]

class StoryAnalysis(BaseModel):
    issues: list[str]          # list of short descriptions of issues/inconsistencies
    thematic_alignment: str    # analysis of how well the theme is conveyed

class SceneScript(BaseModel):
    scene_number: int
    content: str               # the scene description or dialogue in screenplay format

class Screenplay(BaseModel):
    scenes: list[SceneScript]

# 2. Initialize the structure's LLM driver (using OpenAI GPT-4)
structure = Workflow(
    prompt_driver=OpenAiChatPromptDriver(
        model="gpt-4",                 # Use GPT-4 for high-quality, schema-guided output
        temperature=0.0,               # Deterministic outputs for consistency
        max_tokens=2000                # Allow sufficient length for detailed outputs
        # (The OpenAI API key is provided via the OPENAI_API_KEY environment variable)
    )
)

# 3. Define the PromptTasks for each agent and add them to the workflow
plot_architect_task = PromptTask(
    # Plot Architect Prompt: generate PlotOutline JSON from premise
    "You are the **Plot Architect**. Given the film premise: '{{ args[0] }}', "
    "create a detailed **PlotOutline** as a JSON object with the following keys:\n"
    "- **theme**: the story's main theme or message\n"
    "- **protagonist_want**: what the protagonist wants initially\n"
    "- **protagonist_need**: what the protagonist truly needs or learns\n"
    "- **scenes**: a list of scenes, each with a **description** and the key **value_change** in that scene\n\n"
    "Output *only* the JSON for the PlotOutline, adhering to the exact schema.",
    id="plot_outline",
    output_schema=PlotOutline,
    child_ids=["characters", "analysis", "screenplay"]
)

character_designer_task = PromptTask(
    # Character Designer Prompt: generate CharacterProfiles JSON from PlotOutline
    "You are the **Character Designer**. Using the PlotOutline JSON below:\n"
    "{{ parent_outputs['plot_outline'] }}\n\n"
    "Develop a **CharacterProfiles** JSON with a list of main characters. For each character, include:\n"
    "- **name**\n- **role** (e.g. protagonist, antagonist)\n- **backstory**\n- **desire** (their goal)\n- **need** (their true need)\n- **arc** (how the character changes)\n\n"
    "Output only the JSON object matching the CharacterProfiles schema.",
    id="characters",
    output_schema=CharacterProfiles,
    child_ids=["analysis", "screenplay"]
)

thematic_analyst_task = PromptTask(
    # Thematic Analyst Prompt: analyze theme and issues using PlotOutline + CharacterProfiles
    "You are the **Thematic Analyst**. Review the story outline and characters below:\n"
    "**Plot Outline:** {{ parent_outputs['plot_outline'] }}\n"
    "**Characters:** {{ parent_outputs['characters'] }}\n\n"
    "Provide a **StoryAnalysis** JSON object evaluating the story. Include:\n"
    "- **issues**: a list of any plot or character issues/inconsistencies you see\n"
    "- **thematic_alignment**: a brief discussion of how well the story's theme is conveyed, and any suggestions to improve it\n\n"
    "Only output the JSON object matching the StoryAnalysis schema.",
    id="analysis",
    output_schema=StoryAnalysis
    # (No child_ids since this is a terminal task for analysis)
)

scene_shaper_task = PromptTask(
    # Scene Shaper Prompt: create Screenplay JSON (scenes) from PlotOutline + CharacterProfiles
    "You are the **Scene Shaper**. Based on the Plot Outline and Characters below, write a screenplay outline:\n"
    "**Plot Outline:** {{ parent_outputs['plot_outline'] }}\n"
    "**Characters:** {{ parent_outputs['characters'] }}\n\n"
    "Produce a **Screenplay** JSON object with a list of scenes. Each scene should have:\n"
    "- **scene_number**: the scene index\n- **content**: the scene description or sample dialogue (in screenplay style, e.g., with setting or character dialogue)\n\n"
    "Output only the JSON object following the Screenplay schema.",
    id="screenplay",
    output_schema=Screenplay
    # (No child_ids since this is the final screenplay output)
)

# Add all tasks to the workflow structure
structure.add_tasks(
    plot_architect_task,
    character_designer_task,
    thematic_analyst_task,
    scene_shaper_task
)

# The structure will expect an input (premise) when run. 
# Each task's output will be logged and can be retrieved by its id for debugging or further use.
```python
