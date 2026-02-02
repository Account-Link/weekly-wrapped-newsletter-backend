"""Add weekly_trend_hashtag, weekly_trend_sound, weekly_trend_creator tables.

Revision ID: 0011_weekly_trend_tables
Revises: 0010_weekly_report_global
Create Date: 2026-02-02

"""

from alembic import op
import sqlalchemy as sa


revision = "0011_weekly_trend_tables"
down_revision = "0010_weekly_report_global"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "weekly_trend_hashtag",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("global_report_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("hashtag_id", sa.String(), nullable=True),
        sa.Column("hashtag_name", sa.String(), nullable=True),
        sa.Column("country_code", sa.String(), nullable=True),
        sa.Column("publish_cnt", sa.Integer(), nullable=True),
        sa.Column("video_views", sa.Integer(), nullable=True),
        sa.Column("rank_diff", sa.Integer(), nullable=True),
        sa.Column("rank_diff_type", sa.Integer(), nullable=True),
        sa.Column("trend", sa.JSON(), nullable=True),
        sa.Column("industry_info", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weekly_trend_hashtag_global_report_id", "weekly_trend_hashtag", ["global_report_id"], unique=False)
    op.create_index("ix_weekly_trend_hashtag_global_rank", "weekly_trend_hashtag", ["global_report_id", "rank"], unique=False)

    op.create_table(
        "weekly_trend_sound",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("global_report_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("clip_id", sa.String(), nullable=True),
        sa.Column("song_id", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("author", sa.String(), nullable=True),
        sa.Column("country_code", sa.String(), nullable=True),
        sa.Column("duration", sa.Integer(), nullable=True),
        sa.Column("link", sa.String(), nullable=True),
        sa.Column("trend", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weekly_trend_sound_global_report_id", "weekly_trend_sound", ["global_report_id"], unique=False)
    op.create_index("ix_weekly_trend_sound_global_rank", "weekly_trend_sound", ["global_report_id", "rank"], unique=False)

    op.create_table(
        "weekly_trend_creator",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("global_report_id", sa.Integer(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("tcm_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("nick_name", sa.String(), nullable=True),
        sa.Column("avatar_url", sa.String(), nullable=True),
        sa.Column("country_code", sa.String(), nullable=True),
        sa.Column("follower_cnt", sa.Integer(), nullable=True),
        sa.Column("liked_cnt", sa.Integer(), nullable=True),
        sa.Column("tt_link", sa.String(), nullable=True),
        sa.Column("items", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weekly_trend_creator_global_report_id", "weekly_trend_creator", ["global_report_id"], unique=False)
    op.create_index("ix_weekly_trend_creator_global_rank", "weekly_trend_creator", ["global_report_id", "rank"], unique=False)
    op.create_index("ix_weekly_trend_creator_user_id", "weekly_trend_creator", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_weekly_trend_creator_user_id", table_name="weekly_trend_creator")
    op.drop_index("ix_weekly_trend_creator_global_rank", table_name="weekly_trend_creator")
    op.drop_index("ix_weekly_trend_creator_global_report_id", table_name="weekly_trend_creator")
    op.drop_table("weekly_trend_creator")

    op.drop_index("ix_weekly_trend_sound_global_rank", table_name="weekly_trend_sound")
    op.drop_index("ix_weekly_trend_sound_global_report_id", table_name="weekly_trend_sound")
    op.drop_table("weekly_trend_sound")

    op.drop_index("ix_weekly_trend_hashtag_global_rank", table_name="weekly_trend_hashtag")
    op.drop_index("ix_weekly_trend_hashtag_global_report_id", table_name="weekly_trend_hashtag")
    op.drop_table("weekly_trend_hashtag")
