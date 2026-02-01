"""Add weekly_report_global table and batch processing columns to weekly_report.

Revision ID: 0010_weekly_report_global
Revises: 0009_weekly_report_period_dates
Create Date: 2026-01-31
"""

from alembic import op
import sqlalchemy as sa


revision = "0010_weekly_report_global"
down_revision = "0009_weekly_report_period_dates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create weekly_report_global table
    op.create_table(
        "weekly_report_global",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("period_start", sa.DateTime(), nullable=False),
        sa.Column("period_end", sa.DateTime(), nullable=False),
        sa.Column("total_users", sa.Integer(), nullable=True),
        sa.Column("total_videos", sa.Integer(), nullable=True),
        sa.Column("total_watch_hours", sa.Float(), nullable=True),
        sa.Column("analysis_result", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weekly_report_global_period_start", "weekly_report_global", ["period_start"], unique=False)
    op.create_index("ix_weekly_report_global_status", "weekly_report_global", ["status"], unique=False)

    # Add new columns to weekly_report table
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("weekly_report")}

    if "global_report_id" not in cols:
        op.add_column("weekly_report", sa.Column("global_report_id", sa.Integer(), nullable=True))
        op.create_index("ix_weekly_report_global_report_id", "weekly_report", ["global_report_id"], unique=False)

    if "fetch_status" not in cols:
        op.add_column(
            "weekly_report",
            sa.Column("fetch_status", sa.String(), nullable=False, server_default="pending"),
        )

    if "analyze_status" not in cols:
        op.add_column(
            "weekly_report",
            sa.Column("analyze_status", sa.String(), nullable=False, server_default="pending"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("weekly_report")}

    if "analyze_status" in cols:
        op.drop_column("weekly_report", "analyze_status")
    if "fetch_status" in cols:
        op.drop_column("weekly_report", "fetch_status")
    if "global_report_id" in cols:
        op.drop_index("ix_weekly_report_global_report_id", table_name="weekly_report")
        op.drop_column("weekly_report", "global_report_id")

    op.drop_index("ix_weekly_report_global_status", table_name="weekly_report_global")
    op.drop_index("ix_weekly_report_global_period_start", table_name="weekly_report_global")
    op.drop_table("weekly_report_global")
