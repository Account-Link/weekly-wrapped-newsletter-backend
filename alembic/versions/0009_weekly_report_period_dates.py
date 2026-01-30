"""Add period_start and period_end to weekly_report.

Revision ID: 0009_weekly_report_period_dates
Revises: 0008_weekly_report
Create Date: 2026-01-29
"""

from alembic import op
import sqlalchemy as sa


revision = "0009_weekly_report_period_dates"
down_revision = "0008_weekly_report"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("weekly_report")}

    if "period_start" not in cols:
        op.add_column("weekly_report", sa.Column("period_start", sa.DateTime(), nullable=True))
    if "period_end" not in cols:
        op.add_column("weekly_report", sa.Column("period_end", sa.DateTime(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("weekly_report")}

    if "period_end" in cols:
        op.drop_column("weekly_report", "period_end")
    if "period_start" in cols:
        op.drop_column("weekly_report", "period_start")
