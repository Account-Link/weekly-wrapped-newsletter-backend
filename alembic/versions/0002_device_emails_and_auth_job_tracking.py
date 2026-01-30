"""Add device_emails and enrich app_auth_jobs for redirect-driven finalize.

Revision ID: 0002_device_emails
Revises: 0001_initial
Create Date: 2025-12-19
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_device_emails"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NOTE: This migration may have been partially applied in production (e.g. if a prior
    # revision string was too long for `alembic_version.version_num`). Make this upgrade
    # idempotent so `alembic upgrade head` can be safely re-run.
    bind = op.get_bind()
    insp = sa.inspect(bind)

    tables = set(insp.get_table_names())
    if "device_emails" not in tables:
        op.create_table(
            "device_emails",
            sa.Column("device_id", sa.String(), primary_key=True),
            sa.Column("email", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )

    try:
        device_email_indexes = {idx.get("name") for idx in insp.get_indexes("device_emails")}
    except Exception:
        device_email_indexes = set()
    if "ix_device_emails_email" not in device_email_indexes:
        op.create_index("ix_device_emails_email", "device_emails", ["email"], unique=False)

    auth_cols = {col.get("name") for col in insp.get_columns("app_auth_jobs")}
    if "updated_at" not in auth_cols:
        op.add_column("app_auth_jobs", sa.Column("updated_at", sa.DateTime(), nullable=True))
    if "email" not in auth_cols:
        op.add_column("app_auth_jobs", sa.Column("email", sa.String(), nullable=True))
    if "platform" not in auth_cols:
        op.add_column("app_auth_jobs", sa.Column("platform", sa.String(), nullable=True))
    if "app_version" not in auth_cols:
        op.add_column("app_auth_jobs", sa.Column("app_version", sa.String(), nullable=True))
    if "os_version" not in auth_cols:
        op.add_column("app_auth_jobs", sa.Column("os_version", sa.String(), nullable=True))
    if "finalized_at" not in auth_cols:
        op.add_column("app_auth_jobs", sa.Column("finalized_at", sa.DateTime(), nullable=True))
    if "session_id" not in auth_cols:
        op.add_column("app_auth_jobs", sa.Column("session_id", sa.String(), nullable=True))

    try:
        auth_indexes = {idx.get("name") for idx in insp.get_indexes("app_auth_jobs")}
    except Exception:
        auth_indexes = set()
    if "ix_app_auth_jobs_email" not in auth_indexes:
        op.create_index("ix_app_auth_jobs_email", "app_auth_jobs", ["email"], unique=False)

    # Best-effort backfill updated_at to created_at for existing rows.
    op.execute("UPDATE app_auth_jobs SET updated_at = created_at WHERE updated_at IS NULL")


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    try:
        auth_indexes = {idx.get("name") for idx in insp.get_indexes("app_auth_jobs")}
    except Exception:
        auth_indexes = set()
    if "ix_app_auth_jobs_email" in auth_indexes:
        op.drop_index("ix_app_auth_jobs_email", table_name="app_auth_jobs")

    auth_cols = {col.get("name") for col in insp.get_columns("app_auth_jobs")}
    for col_name in ("session_id", "finalized_at", "os_version", "app_version", "platform", "email", "updated_at"):
        if col_name in auth_cols:
            op.drop_column("app_auth_jobs", col_name)

    try:
        device_indexes = {idx.get("name") for idx in insp.get_indexes("device_emails")}
    except Exception:
        device_indexes = set()
    if "ix_device_emails_email" in device_indexes:
        op.drop_index("ix_device_emails_email", table_name="device_emails")

    tables = set(insp.get_table_names())
    if "device_emails" in tables:
        op.drop_table("device_emails")
