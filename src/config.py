from pathlib import Path


def setup():
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    Config.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    PROJECT_DIR = Path(__file__).parents[1]
    DATA_DIR = PROJECT_DIR / "data"
    EXPERIMENT_DIR = PROJECT_DIR / "experiments"


if __name__ == "__main__":
    setup()
