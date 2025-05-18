from griptape.structures import Workflow
from griptape.tasks import PromptTask

def build_workflow():
    workflow = Workflow()

    # TASK 1: Plot Architect
    plot_task = workflow.add_task(build_plot_architect(), input={"premise": "{{ args[0] }}"})

    # TASK 2: Character Designer
    char_task = workflow.add_task(
        build_character_designer(),
        input="{{ tasks.plot_architect.output | to_json }}",
        parent_task_id=plot_task.id
    )

    # TASK 3: Thematic Analyst
    theme_task = workflow.add_task(
        build_thematic_analyst(),
        input={
            "outline": "{{ tasks.plot_architect.output | to_json }}",
            "characters": "{{ tasks.character_designer.output.characters | to_json }}"
        },
        parent_task_id=char_task.id
    )

    # TASK 4: Scene Shaper
    scene_task = workflow.add_task(
        build_scene_shaper(),
        input={
            "premise": "{{ args[0] | to_json }}",
            "outline": "{{ tasks.plot_architect.output | to_json }}",
            "characters": "{{ tasks.character_designer.output.characters | to_json }}",
            "notes": "{{ tasks.thematic_analyst.output.notes | to_json }}"
        },
        parent_task_id=theme_task.id
    )

    workflow.output_task_id = scene_task.id
    return workflow
