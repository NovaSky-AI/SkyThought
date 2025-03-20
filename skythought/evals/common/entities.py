import copy
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionReasoningEffort
from pydantic import BaseModel, ConfigDict, Field
from vllm import SamplingParams as VLLMSamplingParams

TEMPERATURE_DEFAULT = 0
TOP_P_DEFAULT = 1
MAX_TOKENS_DEFAULT = 32768


class Backend(str, Enum):
    VLLM = "vllm"
    OPENAI = "openai"
    RAY = "ray"


class OpenAISamplingParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    temperature: float = TEMPERATURE_DEFAULT
    top_p: float = TOP_P_DEFAULT
    n: int = 1
    max_tokens: int = MAX_TOKENS_DEFAULT
    reasoning_effort: Union[ChatCompletionReasoningEffort, NotGiven] = NOT_GIVEN
    frequency_penalty: Optional[float] = None


class SamplingParameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: Union[OpenAISamplingParams, VLLMSamplingParams]

    @classmethod
    def from_dict(cls, backend: Backend, params: dict):
        params = copy.deepcopy(params)
        if backend == Backend.OPENAI:
            return cls(params=OpenAISamplingParams(**params))
        # Currently, ray-data based processor only supports vllm as the inference engine
        elif backend in [Backend.VLLM, Backend.RAY]:
            return cls(params=VLLMSamplingParams(**params))
        else:
            raise ValueError(f"Invalid backend type: {backend}")

    def __repr__(self):
        return f"SamplingParameters(params={self.params})"

    def to_dict(self):
        if isinstance(self.params, OpenAISamplingParams):
            return self.params.model_dump()
        elif isinstance(self.params, VLLMSamplingParams):
            return {k: getattr(self.params, k) for k in self.params.__annotations__}
        else:
            raise ValueError(f"Invalid sampling parameters type: {type(self.params)}")


class OpenAIClientArgs(BaseModel):
    api_key: Optional[str] = Field(None, description="OpenAI API key")
    base_url: Optional[str] = Field(None, description="OpenAI base URL")
    project: Optional[str] = Field(None, description="OpenAI project")
    organization: Optional[str] = Field(None, description="OpenAI organization")


class RayLLMEngineArgs(BaseModel):

    tensor_parallel_size: Optional[int] = Field(
        default=None, description="Tensor parallelism size"
    )
    num_replicas: Optional[int] = Field(
        default=None, description="Number of replicas to use for Ray"
    )
    batch_size: Optional[int] = Field(default=None, description="Batch size for Ray")
    accelerator_type: Optional[str] = Field(
        default=None, description="Accelerator type for the inference engine"
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=None, description="GPU memory utilization for the inference engine"
    )
    dtype: Optional[Literal["float32", "float16", "bfloat16", "float8", "auto"]] = (
        Field(default=None, description="Data type for inference engine.")
    )

    def get_ray_llm_config(self):
        config_path = Path(
            resources.files("skythought.evals").joinpath("ray_configs/ray_config.yaml")
        )
        with open(config_path) as f:
            default_config = yaml.safe_load(f)

        if self.tensor_parallel_size is not None:
            default_config["engine_kwargs"][
                "tensor_parallel_size"
            ] = self.tensor_parallel_size

        if self.num_replicas is not None:
            default_config["env_config"]["num_replicas"] = self.num_replicas

        if self.batch_size is not None:
            default_config["env_config"]["batch_size"] = self.batch_size

        if self.accelerator_type is not None:
            default_config["accelerator_type"] = self.accelerator_type

        if self.gpu_memory_utilization is not None:
            default_config["engine_kwargs"][
                "gpu_memory_utilization"
            ] = self.gpu_memory_utilization

        # FIXME (sumanthrh): there can be a corner case when we support providing a config yaml directly, and this will override the dtype
        if self.dtype is not None:
            default_config["engine_kwargs"]["dtype"] = self.dtype

        return default_config


@dataclass
class BackendParameters:
    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: Union[dict, OpenAIClientArgs, RayLLMEngineArgs]

    @classmethod
    def from_dict(cls, backend_type: Backend, params: dict):
        if backend_type == Backend.RAY:
            return cls(params=RayLLMEngineArgs(**params))
        elif backend_type == Backend.VLLM:
            # passed directly to LLM(..) instantiation
            return cls(params=params)
        elif backend_type == Backend.OPENAI:
            return cls(params=OpenAIClientArgs(**params))
        else:
            raise ValueError(f"Invalid backend type: {backend_type}")

    def to_dict(self):
        if isinstance(self.params, RayLLMEngineArgs):
            return self.params.model_dump()
        elif isinstance(self.params, dict):
            return self.params
        elif isinstance(self.params, OpenAIClientArgs):
            return self.params.model_dump()
        else:
            raise ValueError(f"Invalid backend parameters type: {type(self.params)}")
