"""Initial schema (consolidated).

Revision ID: 0001_initial
Revises: None
Create Date: 2025-03-15
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_users",
        sa.Column("app_user_id", sa.String(), primary_key=True),
        sa.Column("archive_user_id", sa.String(), nullable=True),
        sa.Column("latest_anchor_token", sa.String(), nullable=True),
        sa.Column("latest_sec_user_id", sa.String(), nullable=True),
        sa.Column("platform_username", sa.String(), nullable=True),
        sa.Column("is_watch_history_available", sa.String(), nullable=False, server_default="unknown"),
        sa.Column("time_zone", sa.String(), nullable=True),
        sa.Column("waitlist_opt_in", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("waitlist_opt_in_at", sa.DateTime(), nullable=True),
        sa.Column("referred_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_app_users_app_user_id", "app_users", ["app_user_id"], unique=False)
    op.create_index("ix_app_users_referred_by", "app_users", ["referred_by"], unique=False)

    op.create_table(
        "app_user_emails",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("app_user_id", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("verified_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_app_user_emails_app_user_id", "app_user_emails", ["app_user_id"], unique=False)
    op.create_index("ix_app_user_emails_email", "app_user_emails", ["email"], unique=False)

    op.create_table(
        "app_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("app_user_id", sa.String(), nullable=False),
        sa.Column("device_id", sa.String(), nullable=False),
        sa.Column("platform", sa.String(), nullable=True),
        sa.Column("app_version", sa.String(), nullable=True),
        sa.Column("os_version", sa.String(), nullable=True),
        sa.Column("token_hash", sa.String(), nullable=False),
        sa.Column("token_encrypted", sa.String(), nullable=False),
        sa.Column("issued_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_app_sessions_app_user_id", "app_sessions", ["app_user_id"], unique=False)
    op.create_index("ix_app_sessions_device_id", "app_sessions", ["device_id"], unique=False)
    op.create_index("ix_app_sessions_token_hash", "app_sessions", ["token_hash"], unique=False)

    op.create_table(
        "app_auth_jobs",
        sa.Column("archive_job_id", sa.String(), primary_key=True),
        sa.Column("app_user_id", sa.String(), nullable=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("device_id", sa.String(), nullable=True),
    )
    op.create_index("ix_app_auth_jobs_app_user_id", "app_auth_jobs", ["app_user_id"], unique=False)

    op.create_table(
        "app_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_name", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("not_before", sa.DateTime(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=True),
        sa.Column("locked_by", sa.String(), nullable=True),
        sa.Column("locked_at", sa.DateTime(), nullable=True),
        sa.Column("lease_seconds", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_app_jobs_task_name", "app_jobs", ["task_name"], unique=False)
    op.create_index("ix_app_jobs_idempotency_key", "app_jobs", ["idempotency_key"], unique=False)

    op.create_table(
        "app_wrapped_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("app_user_id", sa.String(), nullable=False),
        sa.Column("sec_user_id", sa.String(), nullable=False),
        sa.Column("archive_user_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("token_hash", sa.String(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("watch_history_job_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_app_wrapped_runs_app_user_id", "app_wrapped_runs", ["app_user_id"], unique=False)
    op.create_index("ix_app_wrapped_runs_sec_user_id", "app_wrapped_runs", ["sec_user_id"], unique=False)
    op.create_index("ix_app_wrapped_runs_archive_user_id", "app_wrapped_runs", ["archive_user_id"], unique=False)

    op.create_table(
        "referrals",
        sa.Column("code", sa.String(), nullable=False),
        sa.Column("referrer_app_user_id", sa.String(), nullable=True),
        sa.Column("impressions", sa.Integer(), nullable=True, server_default=sa.text("0")),
        sa.Column("conversions", sa.Integer(), nullable=True, server_default=sa.text("0")),
        sa.Column("completions", sa.Integer(), nullable=True, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("code"),
    )
    op.create_index("ix_referrals_referrer_app_user_id", "referrals", ["referrer_app_user_id"], unique=False)

    op.create_table(
        "referral_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("referrer_app_user_id", sa.String(), nullable=True),
        sa.Column("referred_app_user_id", sa.String(), nullable=True),
        sa.Column("event_type", sa.String(), nullable=True),
        sa.Column("archive_job_id", sa.String(), nullable=True),
        sa.Column("wrapped_run_id", sa.String(), nullable=True),
        sa.Column("ip", sa.String(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_referral_events_referrer_app_user_id", "referral_events", ["referrer_app_user_id"], unique=False)
    op.create_index("ix_referral_events_referred_app_user_id", "referral_events", ["referred_app_user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_referral_events_referred_app_user_id", table_name="referral_events")
    op.drop_index("ix_referral_events_referrer_app_user_id", table_name="referral_events")
    op.drop_table("referral_events")

    op.drop_index("ix_referrals_referrer_app_user_id", table_name="referrals")
    op.drop_table("referrals")

    op.drop_index("ix_app_wrapped_runs_archive_user_id", table_name="app_wrapped_runs")
    op.drop_index("ix_app_wrapped_runs_sec_user_id", table_name="app_wrapped_runs")
    op.drop_index("ix_app_wrapped_runs_app_user_id", table_name="app_wrapped_runs")
    op.drop_table("app_wrapped_runs")

    op.drop_index("ix_app_jobs_task_name", table_name="app_jobs")
    op.drop_index("ix_app_jobs_idempotency_key", table_name="app_jobs")
    op.drop_table("app_jobs")

    op.drop_index("ix_app_auth_jobs_app_user_id", table_name="app_auth_jobs")
    op.drop_table("app_auth_jobs")

    op.drop_index("ix_app_sessions_token_hash", table_name="app_sessions")
    op.drop_index("ix_app_sessions_device_id", table_name="app_sessions")
    op.drop_index("ix_app_sessions_app_user_id", table_name="app_sessions")
    op.drop_table("app_sessions")

    op.drop_index("ix_app_user_emails_email", table_name="app_user_emails")
    op.drop_index("ix_app_user_emails_app_user_id", table_name="app_user_emails")
    op.drop_table("app_user_emails")

    op.drop_index("ix_app_users_referred_by", table_name="app_users")
    op.drop_index("ix_app_users_app_user_id", table_name="app_users")
    op.drop_table("app_users")
