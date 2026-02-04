"""Store tiktok radar headers per trend type.

Revision ID: 0013_tiktok_radar_typed_headers
Revises: 0012_tiktok_radar_header_config
Create Date: 2026-02-03

"""

from alembic import op
import sqlalchemy as sa


revision = "0013_tiktok_radar_typed_headers"
down_revision = "0012_tiktok_radar_header_config"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("tiktok_radar_header_config", sa.Column("creator_cookie", sa.Text(), nullable=True))
    op.add_column("tiktok_radar_header_config", sa.Column("creator_user_sign", sa.String(), nullable=True))
    op.add_column("tiktok_radar_header_config", sa.Column("creator_web_id", sa.String(), nullable=True))

    op.add_column("tiktok_radar_header_config", sa.Column("sound_cookie", sa.Text(), nullable=True))
    op.add_column("tiktok_radar_header_config", sa.Column("sound_user_sign", sa.String(), nullable=True))
    op.add_column("tiktok_radar_header_config", sa.Column("sound_web_id", sa.String(), nullable=True))

    op.add_column("tiktok_radar_header_config", sa.Column("hashtag_cookie", sa.Text(), nullable=True))
    op.add_column("tiktok_radar_header_config", sa.Column("hashtag_user_sign", sa.String(), nullable=True))
    op.add_column("tiktok_radar_header_config", sa.Column("hashtag_web_id", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("tiktok_radar_header_config", "hashtag_web_id")
    op.drop_column("tiktok_radar_header_config", "hashtag_user_sign")
    op.drop_column("tiktok_radar_header_config", "hashtag_cookie")

    op.drop_column("tiktok_radar_header_config", "sound_web_id")
    op.drop_column("tiktok_radar_header_config", "sound_user_sign")
    op.drop_column("tiktok_radar_header_config", "sound_cookie")

    op.drop_column("tiktok_radar_header_config", "creator_web_id")
    op.drop_column("tiktok_radar_header_config", "creator_user_sign")
    op.drop_column("tiktok_radar_header_config", "creator_cookie")
