from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    magenta_model: str = "mrt2_small"
    output_dir: str = "./data"
    log_level: str = "INFO"
