# structure.py
from pydantic import BaseModel, Field
from typing import List, Optional

class Scene(BaseModel):
    ...

class CharacterProfile(BaseModel):
    ...

class SceneScript(BaseModel):
    ...

class StoryContext(BaseModel):  # 👈 Unifying Schema
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
