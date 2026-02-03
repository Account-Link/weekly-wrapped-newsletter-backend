"""Add summary_snapshot to weekly_report.

Revision ID: 0011_weekly_report_summary_snapshot
Revises: 0010_weekly_report_global
Create Date: 2026-02-02

"""

from alembic import op
import sqlalchemy as sa


revision = "0011_weekly_report_summary_snapshot"
down_revision = "0010_weekly_report_global"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "weekly_report",
        sa.Column("summary_snapshot", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("weekly_report", "summary_snapshot")
