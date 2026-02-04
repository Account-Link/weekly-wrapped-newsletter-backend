import os
from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env eagerly so uvicorn/gunicorn picks up values.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    archive_base_url: AnyHttpUrl = Field(..., alias="ARCHIVE_BASE_URL")
    archive_api_key: SecretStr = Field(..., alias="ARCHIVE_API_KEY")
    secret_key: SecretStr = Field(..., alias="SECRET_KEY")
    admin_api_key: Optional[SecretStr] = Field(None, alias="ADMIN_API_KEY")

    database_url: Optional[str] = Field(None, alias="DATABASE_URL")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    session_ttl_days: int = Field(30, alias="SESSION_TTL_DAYS")
    port: int = Field(8081, alias="PORT")
    cors_allow_origins: str = Field("*", alias="CORS_ALLOW_ORIGINS")
    weekly_token: Optional[SecretStr] = Field(None, alias="WEEKLY_TOKEN")
    tiktok_trends_admin_password: Optional[SecretStr] = Field(None, alias="TIKTOK_TRENDS_ADMIN_PASSWORD")
    s3_bucket: Optional[str] = Field(None, alias="S3_BUCKET")
    s3_upload_prefix: str = Field("uploads", alias="S3_UPLOAD_PREFIX")
    s3_url: Optional[str] = Field(None, alias="S3_URL")  # Base URL for public links (e.g. CloudFront)
    aws_region: Optional[str] = Field(None, alias="AWS_REGION")

    # TikTok Ads Creative Radar (session-based; cookies/signatures may need manual refresh)
    tiktok_ads_cookie: Optional[str] = Field(None, alias="TIKTOK_ADS_COOKIE")
    tiktok_ads_user_sign: Optional[str] = Field(None, alias="TIKTOK_ADS_USER_SIGN")
    tiktok_ads_web_id: Optional[str] = Field(None, alias="TIKTOK_ADS_WEB_ID")
    tiktok_ads_country_code: str = Field("US", alias="TIKTOK_ADS_COUNTRY_CODE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]


def cors_origins_list(settings: Settings) -> List[str]:
    raw = settings.cors_allow_origins
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]
