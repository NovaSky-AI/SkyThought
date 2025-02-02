import warnings
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

CONFIG_FILE_PATH = Path(__file__).parent / "model_configs.yaml"
# cache the configs in a global var
ALL_CONFIGS = None


class StringInFile(BaseModel):
    path: str
    _string: str = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_and_extract_string(self):
        full_path = Path(CONFIG_FILE_PATH).parent / self.path
        if full_path.exists():
            with open(full_path, "r") as f:
                self._string = f.read()
        else:
            raise ValueError("Invalid path")
        return self

    @property
    def string(self):
        return self._string


def read_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ModelConfig(BaseModel):
    model_id: str
    name: str = Field(default="")
    # can be a string or a path to a file with the string
    system_prompt: Optional[Union[str, StringInFile]] = None
    user_template: Optional[Union[str, StringInFile]] = None

    @field_validator("name", mode="before")
    def validate_name(cls, v):
        if v is None:
            return cls.model_id.split("/")[-1]
        return v

    @classmethod
    def from_model_id(cls, model_id: str):
        global ALL_CONFIGS
        if ALL_CONFIGS is None:
            ALL_CONFIGS = read_yaml(CONFIG_FILE_PATH)
        if model_id in ALL_CONFIGS:
            init_kwargs = ALL_CONFIGS[model_id]
            init_kwargs["model_id"] = model_id
        else:
            init_kwargs = {}
            init_kwargs["model_id"] = model_id
            warnings.warn(
                f"Model {model_id} not found in {CONFIG_FILE_PATH}. Initializing without any system prompt.",
                stacklevel=2,
            )
        return cls(**init_kwargs)
