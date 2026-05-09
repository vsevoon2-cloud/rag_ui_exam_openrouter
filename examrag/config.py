import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_config_dir, user_data_dir


APP_NAME = "ExamRAG"
APP_AUTHOR = "examrag"


def data_dir() -> Path:
    return Path(user_data_dir(APP_NAME, APP_AUTHOR))


def cache_dir() -> Path:
    return Path(user_cache_dir(APP_NAME, APP_AUTHOR))


def config_dir() -> Path:
    return Path(user_config_dir(APP_NAME, APP_AUTHOR))


def ensure_dirs() -> None:
    for p in [data_dir(), cache_dir(), config_dir()]:
        p.mkdir(parents=True, exist_ok=True)


def config_path() -> Path:
    return config_dir() / "config.json"


@dataclass
class AppConfig:
    openrouter_api_key: str = ""
    ocr_model: str = "google/gemini-2.0-flash-001"
    answer_model: str = "google/gemini-2.5-pro"
    embed_model: str = "intfloat/multilingual-e5-large"
    chunking_mode: str = "page"
    retrieval_mode: str = "hybrid"  # "vector" | "hybrid"
    retrieval_retry_count: str = "1"  # "1" | "2" | "3"
    reverify_count: str = "1"  # "1".."5"
    ocr_calls_mode: str = "two_step"  # "two_step" | "single_step"
    last_pdf_path: str = ""
    ocr_max_tokens: str = "1400"
    answer_max_tokens: str = "900"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AppConfig":
        cfg = AppConfig()
        for k in cfg.__dict__.keys():
            if k in d and isinstance(d[k], str):
                setattr(cfg, k, d[k])
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_config() -> AppConfig:
    ensure_dirs()
    p = config_path()
    if not p.exists():
        return AppConfig()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return AppConfig.from_dict(data)
    except Exception:
        pass
    return AppConfig()


def save_config(cfg: AppConfig) -> None:
    ensure_dirs()
    config_path().write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
