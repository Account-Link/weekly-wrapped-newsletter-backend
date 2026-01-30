import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Tuple

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.crypto import encrypt
from app.models import AppSession


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def parse_bearer(auth_header: str) -> str:
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer")
    return auth_header.split(" ", 1)[1]


class SessionService:
    """DB-backed session service with sliding TTL and device binding."""

    def __init__(self, ttl_days: int, secret_key: str) -> None:
        self.ttl_days = ttl_days
        self.secret_key = secret_key

    def _issue_token(self) -> str:
        return secrets.token_urlsafe(32)

    def create_or_rotate(
        self,
        db: Session,
        app_user_id: str,
        device_id: str,
        platform: str,
        app_version: str,
        os_version: str,
    ) -> Tuple[str, datetime]:
        now = datetime.now(timezone.utc)
        token = self._issue_token()
        token_hash = _hash(token)
        expires_at = now + timedelta(days=self.ttl_days)
        token_encrypted = encrypt(token, self.secret_key)

        # Reuse existing record for (app_user_id, device_id) if present.
        rec = (
            db.query(AppSession)
            .filter(
                AppSession.app_user_id == app_user_id,
                AppSession.device_id == device_id,
                AppSession.revoked_at.is_(None),
            )
            .first()
        )
        if rec:
            rec.token_hash = token_hash
            rec.token_encrypted = token_encrypted
            rec.issued_at = now
            rec.expires_at = expires_at
            rec.platform = platform
            rec.app_version = app_version
            rec.os_version = os_version
        else:
            rec = AppSession(
                id=secrets.token_hex(16),
                app_user_id=app_user_id,
                device_id=device_id,
                platform=platform,
                app_version=app_version,
                os_version=os_version,
                token_hash=token_hash,
                token_encrypted=token_encrypted,
                issued_at=now,
                expires_at=expires_at,
                revoked_at=None,
            )
            db.add(rec)
        db.commit()
        return token, expires_at

    def validate(self, db: Session, token: str, device_id: str) -> AppSession:
        token_hash = _hash(token)
        rec = (
            db.query(AppSession)
            .filter(
                AppSession.token_hash == token_hash,
                AppSession.device_id == device_id,
                AppSession.revoked_at.is_(None),
                AppSession.expires_at > datetime.now(timezone.utc),
            )
            .first()
        )
        if not rec:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_session")

        # sliding TTL
        rec.expires_at = datetime.now(timezone.utc) + timedelta(days=self.ttl_days)
        db.commit()
        return rec
