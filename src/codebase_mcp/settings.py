import json
from dataclasses import asdict, dataclass

from .store import _data_dir

KNOWN_MODELS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_FIELDS = {"embedding_model", "vector_size", "api_key", "api_base"}
_OPTIONAL_FIELDS = {"api_key", "api_base"}


@dataclass
class Settings:
    embedding_model: str = "text-embedding-3-small"
    vector_size: int = 1536
    api_key: str | None = None
    api_base: str | None = None


def _settings_path():
    return _data_dir() / "settings.json"


def load_settings() -> Settings:
    path = _settings_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return Settings()
    return Settings(**{k: v for k, v in data.items() if k in _FIELDS})


def save_settings(s: Settings) -> None:
    _data_dir().mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in asdict(s).items() if k not in _OPTIONAL_FIELDS or v is not None}
    _settings_path().write_text(json.dumps(data, indent=2))


def get_settings() -> Settings:
    return load_settings()


def unset_settings_fields(keys: list[str]) -> None:
    path = _settings_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    for k in keys:
        data.pop(k, None)
    _data_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
