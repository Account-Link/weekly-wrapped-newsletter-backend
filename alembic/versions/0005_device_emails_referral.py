"""Add referred_by to device_emails.

Revision ID: 0005_device_emails_referral
Revises: 0004_alembic_version
Create Date: 2025-12-22
"""

from alembic import op
import sqlalchemy as sa


revision = "0005_device_emails_referral"
down_revision = "0004_alembic_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("device_emails")}

    if "referred_by" not in cols:
        op.add_column("device_emails", sa.Column("referred_by", sa.String(), nullable=True))
        op.create_index("ix_device_emails_referred_by", "device_emails", ["referred_by"])


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("device_emails")}

    if "referred_by" in cols:
        with op.batch_alter_table("device_emails") as batch_op:
            batch_op.drop_index("ix_device_emails_referred_by")
            batch_op.drop_column("referred_by")
