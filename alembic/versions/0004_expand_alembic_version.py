"""Expand alembic_version.version_num length to avoid truncation.

Revision ID: 0004_alembic_version
Revises: 0003_device_emails_waitlist
Create Date: 2025-12-19
"""

from contextlib import suppress

from alembic import op
import sqlalchemy as sa


revision = "0004_alembic_version"
down_revision = "0003_device_emails_waitlist"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "alembic_version" not in set(insp.get_table_names()):
        return

    cols = insp.get_columns("alembic_version")
    col = next((c for c in cols if c.get("name") == "version_num"), None)
    if not col:
        return

    length = None
    with suppress(Exception):
        length = getattr(col.get("type"), "length", None)
    if isinstance(length, int) and length >= 128:
        return

    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=col.get("type"),
        type_=sa.String(length=128),
        existing_nullable=False,
    )


def downgrade() -> None:
    # Best-effort: don't shrink if it could truncate a longer revision string.
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "alembic_version" not in set(insp.get_table_names()):
        return

    cols = insp.get_columns("alembic_version")
    col = next((c for c in cols if c.get("name") == "version_num"), None)
    if not col:
        return

    length = None
    with suppress(Exception):
        length = getattr(col.get("type"), "length", None)
    if not isinstance(length, int) or length <= 32:
        return

    # Only shrink if all stored values fit into 32 chars.
    try:
        rows = bind.execute(sa.text("SELECT max(length(version_num)) FROM alembic_version")).fetchone()
        max_len = int(rows[0] or 0) if rows else 0
    except Exception:
        return
    if max_len > 32:
        return

    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=col.get("type"),
        type_=sa.String(length=32),
        existing_nullable=False,
    )

