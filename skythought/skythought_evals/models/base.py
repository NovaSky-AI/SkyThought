import warnings
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator

MODEL_CONFIG_FILE_PATH = Path(__file__).parent / "model_configs.yaml"
# cache the configs in a global var
ALL_MODEL_CONFIGS = None


class StringInFile(BaseModel):
    path: str
    _string: str = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_and_extract_string(self):
        full_path = Path(MODEL_CONFIG_FILE_PATH).parent / self.path
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
    name: Optional[str] = Field(default=None)
    # can be a string or a path to a file with the string
    system_prompt: Optional[Union[str, StringInFile]] = None
    user_template: Optional[Union[str, StringInFile]] = None
    assistant_prefill: Optional[str] = None

    @model_validator(mode="after")
    def validate_name(self):
        if self.name is None:
            self.name = self.model_id.split("/")[-1]
        return self

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        system_prompt_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
    ):
        global ALL_MODEL_CONFIGS
        # only one of the two can be provided
        assert (
            system_prompt_name is None or system_prompt is None
        ), "Only one of `system_prompt_name` or `system_prompt` can be provided"
        init_kwargs = {}
        if ALL_MODEL_CONFIGS is None:
            ALL_MODEL_CONFIGS = read_yaml(MODEL_CONFIG_FILE_PATH)
        if model_id in ALL_MODEL_CONFIGS["models"]:
            init_kwargs = ALL_MODEL_CONFIGS["models"][model_id]

        if system_prompt_name:
            if system_prompt_name not in ALL_MODEL_CONFIGS["system_prompts"]:
                raise ValueError(
                    f"Invalid system prompt template {system_prompt_name} provided."
                )
            init_kwargs["system_prompt"] = ALL_MODEL_CONFIGS["system_prompts"][
                system_prompt_name
            ]
        elif system_prompt:
            init_kwargs["system_prompt"] = system_prompt
        # if none was provided, and the model is not in the config file
        elif model_id not in ALL_MODEL_CONFIGS["models"]:
            init_kwargs = {}
            warnings.warn(
                f"Model {model_id} not found in {MODEL_CONFIG_FILE_PATH}. Initializing without any system prompt.",
                stacklevel=2,
            )

        if assistant_prefill:
            init_kwargs["assistant_prefill"] = assistant_prefill

        init_kwargs["model_id"] = model_id
        return cls(**init_kwargs)


def get_system_prompt_keys():
    global ALL_MODEL_CONFIGS
    if ALL_MODEL_CONFIGS is None:
        ALL_MODEL_CONFIGS = read_yaml(MODEL_CONFIG_FILE_PATH)
    return list(ALL_MODEL_CONFIGS["system_prompts"].keys())
