"""Add period_start/period_end to weekly trend tables.

Revision ID: 0014_weekly_trend_period_dates
Revises: 0013_tiktok_radar_typed_headers
Create Date: 2026-02-03

"""

from alembic import op
import sqlalchemy as sa


revision = "0014_weekly_trend_period_dates"
down_revision = "0013_tiktok_radar_typed_headers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("weekly_trend_hashtag", sa.Column("period_start", sa.DateTime(), nullable=True))
    op.add_column("weekly_trend_hashtag", sa.Column("period_end", sa.DateTime(), nullable=True))

    op.add_column("weekly_trend_sound", sa.Column("period_start", sa.DateTime(), nullable=True))
    op.add_column("weekly_trend_sound", sa.Column("period_end", sa.DateTime(), nullable=True))

    op.add_column("weekly_trend_creator", sa.Column("period_start", sa.DateTime(), nullable=True))
    op.add_column("weekly_trend_creator", sa.Column("period_end", sa.DateTime(), nullable=True))

    # Backfill from weekly_report_global so existing trend rows can be tied to a week.
    op.execute(
        """
        UPDATE weekly_trend_hashtag
        SET period_start = (
                SELECT period_start
                FROM weekly_report_global
                WHERE weekly_report_global.id = weekly_trend_hashtag.global_report_id
            ),
            period_end = (
                SELECT period_end
                FROM weekly_report_global
                WHERE weekly_report_global.id = weekly_trend_hashtag.global_report_id
            )
        """
    )
    op.execute(
        """
        UPDATE weekly_trend_sound
        SET period_start = (
                SELECT period_start
                FROM weekly_report_global
                WHERE weekly_report_global.id = weekly_trend_sound.global_report_id
            ),
            period_end = (
                SELECT period_end
                FROM weekly_report_global
                WHERE weekly_report_global.id = weekly_trend_sound.global_report_id
            )
        """
    )
    op.execute(
        """
        UPDATE weekly_trend_creator
        SET period_start = (
                SELECT period_start
                FROM weekly_report_global
                WHERE weekly_report_global.id = weekly_trend_creator.global_report_id
            ),
            period_end = (
                SELECT period_end
                FROM weekly_report_global
                WHERE weekly_report_global.id = weekly_trend_creator.global_report_id
            )
        """
    )


def downgrade() -> None:
    op.drop_column("weekly_trend_creator", "period_end")
    op.drop_column("weekly_trend_creator", "period_start")

    op.drop_column("weekly_trend_sound", "period_end")
    op.drop_column("weekly_trend_sound", "period_start")

    op.drop_column("weekly_trend_hashtag", "period_end")
    op.drop_column("weekly_trend_hashtag", "period_start")
