"""
structure.py – Griptape 0.23.x Cloud‐ready
-----------------------------------------
Pipeline tasks:

1. Plot Architect    -> Outline JSON
2. Character Designer-> Characters JSON
3. Thematic Analyst  -> Notes   JSON
4. Scene Shaper      -> Scenes  JSON

After the run, we collect all outputs and save them
to /outputs/final_story.json so Griptape Cloud exposes
the file for download.
"""
import os
import sys
import json
from griptape.structures import Pipeline
from griptape.tasks import PromptTask
from griptape.drivers import OpenAiChatPromptDriver
from griptape.config import (
    StructureConfig,
    StructureGlobalDriversConfig,
)

# ---------------------------------------------------------------------------
# 0‧ Safety checks
# ---------------------------------------------------------------------------
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY must be set in Griptape Cloud env vars")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4")  # change if you prefer gpt-3.5-turbo

# ---------------------------------------------------------------------------
# 1‧ Build the Pipeline with the official 0.23 config
# ---------------------------------------------------------------------------
pipe = Pipeline(
    config=StructureConfig(
        global_drivers=StructureGlobalDriversConfig(
            prompt_driver=OpenAiChatPromptDriver(api_key=API_KEY, model=MODEL, temperature=0.3)
        )
    )
)

# ---------------------------------------------------------------------------
# 2‧ Add the four PromptTasks with your original prompts
# ---------------------------------------------------------------------------

# -- 1 Plot Architect --------------------------------------------------------
pipe.add_task(
    PromptTask(
        id="plot_architect",
        prompt="""
You are a Plot Architect AI. Develop a screenplay outline using Robert McKee's Story structure.

### Context:
Premise: {{ args[0] }}

### Instructions:
Follow the Three-Act structure:

- **Act I (Setup):** Inciting Incident, protagonist Desire & Need.
- **Act II (Conflict):** Rising stakes, midpoint, crisis forcing Need.
- **Act III (Resolution):** Climax & transformation.

For each scene include:
- `act`, `number`, `description`, `conflict`, `value_change`

Also output:
- `title`, `theme`, `protagonist_desire`, `protagonist_need`

### Output:
Respond **ONLY** with a valid JSON object:
{
  "title": "...",
  "theme": "...",
  "protagonist_desire": "...",
  "protagonist_need": "...",
  "scenes": [
    { "act": 1, "number": 1, "description": "...", "conflict": "...", "value_change": "..." }
  ]
}
""".strip()
    )
)

# -- 2 Character Designer ----------------------------------------------------
pipe.add_task(
    PromptTask(
        id="character_designer",
        prompt="""
You are a Character Designer AI.

### Outline:
{{ parent_output }}

### Instructions:
Create 3-5 characters who embody or challenge the theme.
Each character must have:
- `name`, `role`, `backstory`, `desire`, `need`, `arc`

### Output:
Respond **ONLY** with:
{ "characters": [ { ... }, ... ] }
""".strip()
    )
)

# -- 3 Thematic Analyst ------------------------------------------------------
pipe.add_task(
    PromptTask(
        id="thematic_analyst",
        prompt="""
You are a Thematic Analyst AI.

### Outline:
{{ tasks.plot_architect.output }}

### Characters:
{{ tasks.character_designer.output }}

### Instructions:
For each scene check:
- conflict clarity
- value change
- thematic relevance

Also check character arcs.

### Output:
{ "issues_found": true|false, "notes": ["...", "..."] }
""".strip()
    )
)

# -- 4 Scene Shaper ----------------------------------------------------------
pipe.add_task(
    PromptTask(
        id="scene_shaper",
        prompt="""
You are a Scene Shaper AI.

### Premise
{{ args[0] }}

### Outline
{{ tasks.plot_architect.output }}

### Characters
{{ tasks.character_designer.output }}

### Analyst Notes
{{ tasks.thematic_analyst.output }}

### Instructions
Write screenplay scenes (INT/EXT HEADINGS, action, dialogue).
Show conflict and value change; weave the theme via subtext.

### Output
{ "scenes": [ { "number": 1, "content": "Scene text..." }, ... ] }
""".strip(),
        # Slightly higher creativity for prose
        driver=OpenAiChatPromptDriver(api_key=API_KEY, model=MODEL, temperature=0.7),
    )
)

# ---------------------------------------------------------------------------
# 3‧ Execute when run by Griptape Cloud
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Pass along every CLI arg Griptape supplies (the first is premise)
    result = pipe.run(*sys.argv[1:])

    # Collect outputs
    outline_json   = pipe.tasks.plot_architect.output.value
    chars_json     = pipe.tasks.character_designer.output.value
    notes_json     = pipe.tasks.thematic_analyst.output.value
    scenes_json    = pipe.tasks.scene_shaper.output.value

    # Build dict for file
    story_bundle = {
        "outline":        json.loads(outline_json),
        "characters":     json.loads(chars_json),
        "analysis_notes": json.loads(notes_json),
        "scenes":         json.loads(scenes_json)
    }

    # Write to Griptape Cloud /outputs directory
    os.makedirs("/outputs", exist_ok=True)
    with open("/outputs/final_story.json", "w") as fp:
        json.dump(story_bundle, fp, indent=2)

    # Optional console echo
    print("✅ Saved screenplay to /outputs/final_story.json")
