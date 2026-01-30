"""Add redirect click tracking to app_auth_jobs.

Revision ID: 0007_app_auth_jobs_redirect_clicks
Revises: 0006_app_auth_jobs_ip
Create Date: 2026-01-05
"""

from alembic import op
import sqlalchemy as sa


revision = "0007_app_auth_jobs_redirect_clicks"
down_revision = "0006_app_auth_jobs_ip"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("app_auth_jobs")}

    if "redirect_clicked_at" not in cols:
        op.add_column("app_auth_jobs", sa.Column("redirect_clicked_at", sa.DateTime(), nullable=True))
    if "redirect_clicks" not in cols:
        op.add_column(
            "app_auth_jobs",
            sa.Column("redirect_clicks", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("app_auth_jobs")}

    if "redirect_clicks" in cols:
        op.drop_column("app_auth_jobs", "redirect_clicks")
    if "redirect_clicked_at" in cols:
        op.drop_column("app_auth_jobs", "redirect_clicked_at")
