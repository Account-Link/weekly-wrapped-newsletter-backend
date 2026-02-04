"""Add tiktok_radar_header_config table.

Revision ID: 0012_tiktok_radar_header_config
Revises: 0011_weekly_trend_tables
Create Date: 2026-02-03

"""

from alembic import op
import sqlalchemy as sa


revision = "0012_tiktok_radar_header_config"
down_revision = "0011_weekly_trend_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tiktok_radar_header_config",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("cookie", sa.Text(), nullable=True),
        sa.Column("user_sign", sa.String(), nullable=True),
        sa.Column("web_id", sa.String(), nullable=True),
        sa.Column("country_code", sa.String(), nullable=False, server_default="US"),
        sa.Column("updated_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tiktok_radar_header_config_id", "tiktok_radar_header_config", ["id"], unique=False)
    op.create_index("ix_tiktok_radar_header_config_updated_at", "tiktok_radar_header_config", ["updated_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_tiktok_radar_header_config_updated_at", table_name="tiktok_radar_header_config")
    op.drop_index("ix_tiktok_radar_header_config_id", table_name="tiktok_radar_header_config")
    op.drop_table("tiktok_radar_header_config")
