"""S3 upload routes."""
import os
from datetime import datetime
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.schemas import UploadResponse
from app.settings import get_settings
from app.observability import get_logger

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post(
    "/upload",
    response_model=UploadResponse,
    tags=["upload"],
)
async def upload_image(
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload an image file to S3."""
    if not settings.s3_bucket:
        raise HTTPException(status_code=500, detail="s3_not_configured")
    if not settings.s3_url:
        raise HTTPException(status_code=500, detail="s3_url_not_configured")
    
    # Validate content type
    content_type = file.content_type or ""
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if not any(content_type.startswith(t) for t in allowed_types):
        raise HTTPException(status_code=400, detail="invalid_file_type")
    
    # Validate file size (5MB max)
    max_size = 5 * 1024 * 1024  # 5MB
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(status_code=400, detail="file_too_large")
    
    # Determine file extension from content type
    ext_map = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    ext = ext_map.get(content_type.split(";")[0].strip(), "jpg")
    
    # Generate S3 key
    now = datetime.utcnow()
    prefix = settings.s3_upload_prefix or "uploads"
    key = f"{prefix}/{now.year}/{now.month:02d}/{uuid4()}.{ext}"
    
    # Upload to S3
    try:
        s3_client = boto3.client(
            "s3",
            region_name=settings.aws_region or os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_SECRET"),
        )
        
        s3_client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=contents,
            ContentType=content_type,
        )
        
        # Generate public URL
        url = f"{settings.s3_url}/{key}"
        
        return UploadResponse(url=url, key=key)
    except (BotoCoreError, ClientError) as e:
        logger.exception(
            "upload.s3_error",
            extra={"event": "upload.s3_error", "error": str(e)},
        )
        raise HTTPException(status_code=500, detail="upload_failed")
