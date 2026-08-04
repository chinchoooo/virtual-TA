"""Runtime settings (secrets, endpoints, paths), read from the environment / `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    value = os.getenv(env)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    api_base_url: str
    data_dir: Path
    cache_dir: Path
    runs_dir: Path
    request_timeout: float = 60.0
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        origins = os.getenv("CORS_ORIGINS", "*")
        return cls(
            api_key=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_base_url=os.getenv("API_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/"),
            data_dir=_path("VTA_DATA_DIR", PROJECT_ROOT / "data"),
            cache_dir=_path("VTA_CACHE_DIR", PROJECT_ROOT / ".cache"),
            runs_dir=_path("VTA_RUNS_DIR", PROJECT_ROOT / "runs"),
            request_timeout=float(os.getenv("VTA_REQUEST_TIMEOUT", "60")),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "No API key configured. Set API_KEY (or OPENAI_API_KEY) in your environment "
                "or .env file. See .env.example."
            )
        return self.api_key
