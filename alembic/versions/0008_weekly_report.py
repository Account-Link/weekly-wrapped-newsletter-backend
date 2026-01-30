"""Add weekly_report table and app_users.weekly_report_unsubscribed.

Revision ID: 0008_weekly_report
Revises: 0007_app_auth_jobs_redirect_clicks
Create Date: 2026-01-29
"""

from alembic import op
import sqlalchemy as sa


revision = "0008_weekly_report"
down_revision = "0007_app_auth_jobs_redirect_clicks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create weekly_report table
    op.create_table(
        "weekly_report",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("app_user_id", sa.String(), nullable=False),
        sa.Column("email_content", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("send_status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("feeding_state", sa.String(), nullable=True),
        sa.Column("trend_name", sa.String(), nullable=True),
        sa.Column("trend_type", sa.String(), nullable=True),
        sa.Column("discovery_rank", sa.Integer(), nullable=True),
        sa.Column("total_discoverers", sa.Integer(), nullable=True),
        sa.Column("origin_niche_text", sa.String(), nullable=True),
        sa.Column("spread_end_text", sa.String(), nullable=True),
        sa.Column("reach_start", sa.Float(), nullable=True),
        sa.Column("reach_end", sa.Float(), nullable=True),
        sa.Column("current_reach", sa.Float(), nullable=True),
        sa.Column("total_videos", sa.Integer(), nullable=True),
        sa.Column("total_time", sa.Integer(), nullable=True),
        sa.Column("pre_total_time", sa.Integer(), nullable=True),
        sa.Column("miles_scrolled", sa.Integer(), nullable=True),
        sa.Column("topics", sa.JSON(), nullable=True),
        sa.Column("timezone", sa.String(), nullable=True),
        sa.Column("rabbit_hole_datetime", sa.DateTime(), nullable=True),
        sa.Column("rabbit_hole_date", sa.String(), nullable=True),
        sa.Column("rabbit_hole_time", sa.String(), nullable=True),
        sa.Column("rabbit_hole_count", sa.Integer(), nullable=True),
        sa.Column("rabbit_hole_category", sa.String(), nullable=True),
        sa.Column("nudge_text", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weekly_report_app_user_id", "weekly_report", ["app_user_id"], unique=False)
    op.create_index("ix_weekly_report_created_at", "weekly_report", ["created_at"], unique=False)

    # Add weekly_report_unsubscribed to app_users
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("app_users")}

    if "weekly_report_unsubscribed" not in cols:
        op.add_column(
            "app_users",
            sa.Column("weekly_report_unsubscribed", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {col.get("name") for col in insp.get_columns("app_users")}

    if "weekly_report_unsubscribed" in cols:
        op.drop_column("app_users", "weekly_report_unsubscribed")

    op.drop_index("ix_weekly_report_created_at", table_name="weekly_report")
    op.drop_index("ix_weekly_report_app_user_id", table_name="weekly_report")
    op.drop_table("weekly_report")
