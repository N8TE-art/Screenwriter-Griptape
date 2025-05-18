version: "1.0"
runtime: python3
runtime_version: "3.11"

structure:
  id: screenwriter_pipeline
  description: >
    A Griptape structure that chains four prebuilt agents (Plot Architect, Character Designer,
    Thematic Analyst, Scene Shaper) using Robert McKee story principles and a shared StoryContext.

  inputs:
    - name: premise
      description: "Story premise (logline)"
      required: true
      type: string

  tasks:
    - id: plot_architect
      type: agent
      agent_id: a22089b6-420d-4dd3-8aa8-c2689f59eab7
      input_template: |
        {{ args[0] }}

    - id: character_designer
      type: agent
      agent_id: dee8980d-a058-47f4-b3cb-71288f7592de
      input_template: |
        {{ tasks.plot_architect.output | to_json }}

    - id: thematic_analyst
      type: agent
      agent_id: 2bf4393c-437e-4c72-a856-b58cae433e3c
      input_template: |
        {
          "outline": {{ tasks.plot_architect.output | to_json }},
          "characters": {{ tasks.character_designer.output | to_json }}
        }

    - id: scene_shaper
      type: agent
      agent_id: f5187591-9a49-454e-add4-50eec9bc4ca8
      input_template: |
        {
          "premise": {{ args[0] | to_json }},
          "outline": {{ tasks.plot_architect.output | to_json }},
          "characters": {{ tasks.character_designer.output | to_json }},
          "notes": {{ tasks.thematic_analyst.output.notes | to_json }}
        }

  output_task: scene_shaper

run:
  main_file: structure.py
  entrypoint: structure:build_workflow
