from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Barcode Scanner Service"
    app_env: str = "development"
    max_upload_bytes: int = 15 * 1024 * 1024
    allowed_image_types: str = "image/jpeg,image/png,image/webp"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def allowed_content_types(self) -> set[str]:
        return {
            content_type.strip().lower()
            for content_type in self.allowed_image_types.split(",")
            if content_type.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
