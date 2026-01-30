"""Add client_ip to app_auth_jobs.

Revision ID: 0006_app_auth_jobs_ip
Revises: 0005_device_emails_referral
Create Date: 2025-12-30
"""

from alembic import op
import sqlalchemy as sa


revision = "0006_app_auth_jobs_ip"
down_revision = "0005_device_emails_referral"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("app_auth_jobs")}

    if "client_ip" not in cols:
        op.add_column("app_auth_jobs", sa.Column("client_ip", sa.String(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("app_auth_jobs")}

    if "client_ip" in cols:
        op.drop_column("app_auth_jobs", "client_ip")
