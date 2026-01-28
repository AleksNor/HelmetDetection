from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # video
    VIDEO_SOURCE: str | int = 0

    # model
    MODEL_PATH: Path = Path(
        "../app/runs/detect/sh17_person_head_helmet/weights/best.pt"
    )

    # processing
    PROCESS_FPS: int = 25
    VIOLATION_SECONDS: float = 3.0
    COOLDOWN_SECONDS: float = 5.0

    # output
    OUTPUT_DIR: Path = Path("violations")

    # confidence thresholds
    CONF_PERSON: float = 0.35
    CONF_HEAD: float = 0.35
    CONF_HELMET: float = 0.35
    DEVICE: str = "cuda"  # "cpu" | "cuda"

    class Config:
        env_file = ".env"


settings = Settings()

settings.OUTPUT_DIR.mkdir(exist_ok=True)