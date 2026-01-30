"""Add waitlist fields to device_emails (Option A).

Revision ID: 0003_device_emails_waitlist
Revises: 0002_device_emails
Create Date: 2025-12-19
"""

from alembic import op
import sqlalchemy as sa


revision = "0003_device_emails_waitlist"
down_revision = "0002_device_emails"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("device_emails")}

    if "waitlist_opt_in" not in cols:
        op.add_column(
            "device_emails",
            sa.Column("waitlist_opt_in", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    if "waitlist_opt_in_at" not in cols:
        op.add_column("device_emails", sa.Column("waitlist_opt_in_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("device_emails")}

    if "waitlist_opt_in_at" in cols:
        op.drop_column("device_emails", "waitlist_opt_in_at")
    if "waitlist_opt_in" in cols:
        op.drop_column("device_emails", "waitlist_opt_in")
