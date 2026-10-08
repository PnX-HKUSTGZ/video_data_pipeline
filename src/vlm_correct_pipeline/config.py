"""Typed configuration and explicit, non-executing dotenv credential loading."""

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class StrictModel(BaseModel):
    """Reject unknown fields, nonfinite numbers, and scalar coercions."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class APIConfig(StrictModel):
    env_file: str = ".env"
    base_url_env: str = "BASE_URL"
    api_key_env: str = "API_KEY"
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(default=120, gt=0)
    stream: bool = False
    max_output_tokens: int = Field(default=8192, gt=0)

    @field_validator("model")
    @classmethod
    def explicit_model(cls, value: str) -> str:
        if not value.strip() or value == "MODEL_ID_TO_SET":
            raise ValueError("Provide an explicit model identifier")
        return value


class DatasetConfig(StrictModel):
    source: str
    names: dict[int, str] | None = None
    input_task: Literal["detect", "pose_boxes"] = "detect"
    selection_file: str | None = None
    source_rounding_tolerance: float = Field(default=0, ge=0, le=1e-6)


class ReferenceConfig(StrictModel):
    image: str
    label: str
    description: str = ""


class RunConfig(StrictModel):
    image_concurrency: int = Field(default=1, ge=1, le=32)
    max_model_calls: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_consecutive_api_errors: int = Field(default=3, ge=1)


class OutputConfig(StrictModel):
    directory: str


class Config(StrictModel):
    """One reproducible annotation run; paths are relative to its YAML file."""

    api: APIConfig
    dataset: DatasetConfig
    references: list[ReferenceConfig] = Field(min_length=1)
    annotation_rules: str = Field(min_length=1)
    run: RunConfig = Field(default_factory=RunConfig)
    output: OutputConfig
    image_max_edge: int = Field(default=2048, ge=256, le=8192)
    image_format: Literal["jpeg", "png"] = "jpeg"
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    coordinate_grid: bool = False

    @field_validator("annotation_rules")
    @classmethod
    def meaningful_rules(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Annotation rules must not be blank")
        return value


class Credentials(StrictModel):
    base_url: str
    api_key: SecretStr


def local_path(value: str, base: Path) -> Path:
    """Resolve configuration paths independently of the process working directory."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path: Path) -> Config:
    """Parse a local YAML without accessing credentials or the network."""
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def credentials(config: APIConfig, base: Path) -> Credentials:
    """Prefer process variables; dotenv only fills missing values and is never sourced."""
    values = dotenv_values(local_path(config.env_file, base), interpolate=False)
    url = os.environ.get(config.base_url_env, values.get(config.base_url_env))
    key = os.environ.get(config.api_key_env, values.get(config.api_key_env))
    if not url or not key:
        raise ValueError("BASE_URL/API_KEY variables are missing or empty")
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("API base URL must be HTTP(S), without embedded credentials or query")
    return Credentials(base_url=url.rstrip("/"), api_key=SecretStr(key))
