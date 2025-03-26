from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "poromics"
    debug: bool = False
