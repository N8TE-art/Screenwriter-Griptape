import json
import sys
from griptape.structures import Workflow
from griptape.tasks import PromptTask

# Define the multi-agent Workflow structure
class ScreenwritingWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        # Agent 1: Plot Architect - generates logline and outline from premise
        outline_task = PromptTask(
            prompt=(
                "You are a Plot Architect writing a screenplay outline based on the premise: '{{ args[0] }}'.\n"
                "- Provide a one-sentence **Logline** encapsulating the story.\n"
                "- Provide a **Story Outline** as a list of scenes, each with a brief description.\n\n"
                "Respond in JSON with keys 'logline' (string) and 'outline' (array of scene descriptions). "
                "No extra explanation, JSON only."
            ),
            id="outline"
        )
        # Agent 2: Character Designer - creates characters using premise and outline
        characters_task = PromptTask(
            prompt=(
                "You are a Character Designer. Based on the premise '{{ args[0] }}' and the following story outline:\n"
                "{{ parent_outputs['outline'] }}\n\n"
                "List the main characters in JSON format. Include for each: name, role (e.g., protagonist/antagonist), a brief description, "
                "and their character arc or goal. Respond with a JSON array of character objects."
            ),
            id="characters",
            parent_ids=[outline_task.id]
        )
        # Agent 3: Thematic Analyst - determines theme and value transitions using outline & characters
        thematic_task = PromptTask(
            prompt=(
                "You are a Thematic Analyst. Given the story outline:\n{{ parent_outputs['outline'] }}\n"
                "and characters:\n{{ parent_outputs['characters'] }}\n\n"
                "Identify the core theme or value at stake in this story, and the value change in each scene. For each scene in the outline, "
                "specify the dominant value shift (e.g., from 'truth' to 'lie', or 'love' to 'hate').\n"
                "Respond in JSON as an array, where each element has: scene number, from_value, to_value (describing the value transition)."
            ),
            id="theme",
            parent_ids=[outline_task.id, characters_task.id]
        )
        # Agent 4: Scene Shaper - writes full screenplay using outline, characters, and value transitions
        scene_task = PromptTask(
            prompt=(
                "You are a Screenwriter. Write the full screenplay based on the following:\n"
                "- Premise: {{ args[0] }}\n"
                "- Story Outline: {{ parent_outputs['outline'] }}\n"
                "- Characters: {{ parent_outputs['characters'] }}\n"
                "- Scene Value Transitions: {{ parent_outputs['theme'] }}\n\n"
                "The screenplay should be written in a scene-by-scene format with proper scene headings, descriptions, and dialogue. Ensure each scene reflects the specified value change. "
                "Finally, include a one-line Logline at the top and a brief outline and value change summary for reference at the end of the output."
            ),
            id="script",
            parent_ids=[outline_task.id, characters_task.id, thematic_task.id]
        )
        # Add tasks to the workflow
        self.add_task(outline_task)
        self.add_task(characters_task)
        self.add_task(thematic_task)
        self.add_task(scene_task)

# If run as a script (e.g., for local testing), execute the workflow on a given premise
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python structure.py \"<premise>\"")
        sys.exit(1)
    premise = sys.argv[1]
    structure = ScreenwritingWorkflow()
    # Run the workflow with the premise
    structure.run(premise)
    # Retrieve outputs from each task
    outline_text = structure.tasks[0].output.value if structure.tasks[0].output else ""
    characters_text = structure.tasks[1].output.value if structure.tasks[1].output else ""
    theme_text = structure.tasks[2].output.value if structure.tasks[2].output else ""
    screenplay_text = structure.tasks[3].output.value if structure.tasks[3].output else ""
    # Validate and parse JSON outputs
    try:
        outline_data = json.loads(outline_text)
        assert isinstance(outline_data, dict) and "logline" in outline_data and "outline" in outline_data
    except Exception as e:
        outline_data = {}
        print(f"Error: Plot Architect output is invalid JSON or missing keys. ({e})")
    try:
        characters_data = json.loads(characters_text)
        assert isinstance(characters_data, list)
    except Exception as e:
        characters_data = []
        print(f"Error: Character Designer output is invalid JSON. ({e})")
    try:
        transitions_data = json.loads(theme_text)
        assert isinstance(transitions_data, list)
    except Exception as e:
        transitions_data = []
        print(f"Error: Thematic Analyst output is invalid JSON. ({e})")
    # Write each agent's output to a file
    if outline_data:
        with open("plot_outline.json", "w") as f:
            json.dump(outline_data, f, indent=2)
    if characters_data:
        with open("characters.json", "w") as f:
            json.dump(characters_data, f, indent=2)
    if transitions_data:
        with open("value_transitions.json", "w") as f:
            json.dump(transitions_data, f, indent=2)
    if screenplay_text:
        with open("screenplay.txt", "w") as f:
            f.write(screenplay_text)
    # Compose combined output for display (logline, outline, transitions, screenplay)
    combined_output = ""
    if outline_data.get("logline"):
        combined_output += f"LOGLINE: {outline_data['logline']}\n\n"
    if outline_data.get("outline"):
        combined_output += "STORY OUTLINE:\n"
        for scene in outline_data["outline"]:
            combined_output += f"- {scene}\n"
        combined_output += "\n"
    if transitions_data:
        combined_output += "VALUE TRANSITIONS PER SCENE:\n"
        for t in transitions_data:
            if isinstance(t, dict) and "scene" in t:
                from_val = t.get("from_value") or t.get("from") or ""
                to_val = t.get("to_value") or t.get("to") or ""
                combined_output += f"Scene {t.get('scene')}: {from_val} -> {to_val}\n"
        combined_output += "\n"
    combined_output += "SCREENPLAY:\n"
    combined_output += (screenplay_text or "[Screenplay generation failed.]")
    # Print the combined output to console (for local run)
    print(combined_output)
