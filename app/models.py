from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, JSON, String, Text
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class AppUser(Base):
    __tablename__ = "app_users"
    app_user_id = Column(String, primary_key=True, index=True)
    archive_user_id = Column(String, nullable=True)
    latest_anchor_token = Column(String, nullable=True)
    latest_sec_user_id = Column(String, nullable=True)
    platform_username = Column(String, nullable=True)
    is_watch_history_available = Column(String, nullable=False, default="unknown")
    time_zone = Column(String, nullable=True)
    waitlist_opt_in = Column(Boolean, nullable=False, default=False)
    waitlist_opt_in_at = Column(DateTime, nullable=True)
    referred_by = Column(String, nullable=True, index=True)
    weekly_report_unsubscribed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AppUserEmail(Base):
    __tablename__ = "app_user_emails"
    id = Column(String, primary_key=True)
    app_user_id = Column(String, index=True)
    email = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    verified_at = Column(DateTime, nullable=True)


class DeviceEmail(Base):
    __tablename__ = "device_emails"
    device_id = Column(String, primary_key=True)
    email = Column(String, index=True, nullable=False)
    referred_by = Column(String, index=True, nullable=True)
    waitlist_opt_in = Column(Boolean, nullable=False, default=False)
    waitlist_opt_in_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AppSession(Base):
    __tablename__ = "app_sessions"
    id = Column(String, primary_key=True)
    app_user_id = Column(String, index=True)
    device_id = Column(String, index=True)
    platform = Column(String)
    app_version = Column(String)
    os_version = Column(String)
    token_hash = Column(String, index=True)
    token_encrypted = Column(String)
    issued_at = Column(DateTime)
    expires_at = Column(DateTime)
    revoked_at = Column(DateTime, nullable=True)


class AppAuthJob(Base):
    __tablename__ = "app_auth_jobs"
    archive_job_id = Column(String, primary_key=True)
    app_user_id = Column(String, index=True, nullable=True)
    provider = Column(String)  # x_cookie | xordi
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String, default="pending")  # pending|finalizing|finalized|failed|expired
    last_error = Column(Text, nullable=True)
    device_id = Column(String, nullable=True)
    client_ip = Column(String, nullable=True)
    email = Column(String, nullable=True, index=True)
    platform = Column(String, nullable=True)
    app_version = Column(String, nullable=True)
    os_version = Column(String, nullable=True)
    finalized_at = Column(DateTime, nullable=True)
    session_id = Column(String, nullable=True)
    redirect_clicked_at = Column(DateTime, nullable=True)
    redirect_clicks = Column(Integer, default=0)


class AppJob(Base):
    __tablename__ = "app_jobs"
    id = Column(String, primary_key=True)
    task_name = Column(String, index=True)
    payload = Column(JSON)
    status = Column(String, default="pending")  # pending|running|succeeded|failed
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=5)
    not_before = Column(DateTime, nullable=True)
    idempotency_key = Column(String, nullable=True, index=True)
    locked_by = Column(String, nullable=True)
    locked_at = Column(DateTime, nullable=True)
    lease_seconds = Column(Integer, default=60)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AppWrappedRun(Base):
    __tablename__ = "app_wrapped_runs"
    id = Column(String, primary_key=True)
    app_user_id = Column(String, index=True)
    sec_user_id = Column(String, index=True)
    archive_user_id = Column(String, index=True)
    status = Column(String, default="pending")  # pending|ready|failed
    email = Column(String, nullable=True)
    token_hash = Column(String, nullable=True)
    payload = Column(MutableDict.as_mutable(JSON), nullable=True)
    watch_history_job_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Referral(Base):
    __tablename__ = "referrals"
    code = Column(String, primary_key=True)
    referrer_app_user_id = Column(String, index=True)
    impressions = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    completions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReferralEvent(Base):
    __tablename__ = "referral_events"
    id = Column(String, primary_key=True)
    referrer_app_user_id = Column(String, index=True)
    referred_app_user_id = Column(String, index=True, nullable=True)
    event_type = Column(String)  # impression|conversion|completion
    archive_job_id = Column(String, nullable=True)
    wrapped_run_id = Column(String, nullable=True)
    ip = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class WeeklyReportGlobal(Base):
    """Global weekly report for all users - stores aggregated analysis results."""
    __tablename__ = "weekly_report_global"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    period_start = Column(DateTime, nullable=False, index=True)  # 周一 00:00 UTC
    period_end = Column(DateTime, nullable=False)  # 下周一 00:00 UTC
    total_users = Column(Integer, nullable=True)  # 参与分析的用户数
    total_videos = Column(Integer, nullable=True)  # 所有用户观看视频总数
    total_watch_hours = Column(Float, nullable=True)  # 所有用户观看时长总计
    analysis_result = Column(JSON, nullable=True)  # 整体分析结果 (预留, 可存任意分析数据)
    status = Column(String, default="pending", index=True)  # pending|fetching|analyzing|sending|completed|failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WeeklyReport(Base):
    __tablename__ = "weekly_report"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    app_user_id = Column(String, index=True, nullable=False)
    global_report_id = Column(Integer, nullable=True, index=True)  # 关联到 WeeklyReportGlobal
    email_content = Column(Text, nullable=True)
    period_start = Column(DateTime, nullable=True)  # 总结开始时间
    period_end = Column(DateTime, nullable=True)  # 总结结束时间
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    send_status = Column(String, default="pending")  # pending|sent|failed
    fetch_status = Column(String, default="pending")  # pending|fetching|fetched|failed
    analyze_status = Column(String, default="pending")  # pending|analyzing|analyzed|failed
    feeding_state = Column(String, nullable=True)  # curious|excited|cozy|sleepy|dizzy
    trend_name = Column(String, nullable=True)
    trend_type = Column(String, nullable=True)
    discovery_rank = Column(Integer, nullable=True)
    total_discoverers = Column(Integer, nullable=True)
    origin_niche_text = Column(String, nullable=True)
    spread_end_text = Column(String, nullable=True)
    reach_start = Column(Float, nullable=True)
    reach_end = Column(Float, nullable=True)
    current_reach = Column(Float, nullable=True)
    total_videos = Column(Integer, nullable=True)
    total_time = Column(Integer, nullable=True)
    pre_total_time = Column(Integer, nullable=True)
    miles_scrolled = Column(Integer, nullable=True)
    topics = Column(JSON, nullable=True)  # [{topic: string, pic_url: string}]
    timezone = Column(String, nullable=True)
    rabbit_hole_datetime = Column(DateTime, nullable=True)
    rabbit_hole_date = Column(String, nullable=True)
    rabbit_hole_time = Column(String, nullable=True)
    rabbit_hole_count = Column(Integer, nullable=True)
    rabbit_hole_category = Column(String, nullable=True)
    nudge_text = Column(String, nullable=True)


class TikTokRadarHeaderConfig(Base):
    """Manually managed headers for TikTok Creative Radar requests."""
    __tablename__ = "tiktok_radar_header_config"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    cookie = Column(Text, nullable=True)
    user_sign = Column(String, nullable=True)
    web_id = Column(String, nullable=True)
    creator_cookie = Column(Text, nullable=True)
    creator_user_sign = Column(String, nullable=True)
    creator_web_id = Column(String, nullable=True)
    sound_cookie = Column(Text, nullable=True)
    sound_user_sign = Column(String, nullable=True)
    sound_web_id = Column(String, nullable=True)
    hashtag_cookie = Column(Text, nullable=True)
    hashtag_user_sign = Column(String, nullable=True)
    hashtag_web_id = Column(String, nullable=True)
    country_code = Column(String, nullable=False, default="US")
    updated_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WeeklyTrendHashtag(Base):
    """Weekly top 100 hashtags from TikTok Creative Radar for a given global report."""
    __tablename__ = "weekly_trend_hashtag"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    global_report_id = Column(Integer, nullable=False, index=True)
    period_start = Column(DateTime, nullable=True)  # 周一 00:00 UTC
    period_end = Column(DateTime, nullable=True)  # 下周一 00:00 UTC
    rank = Column(Integer, nullable=False)
    hashtag_id = Column(String, nullable=True)
    hashtag_name = Column(String, nullable=True)
    country_code = Column(String, nullable=True)
    publish_cnt = Column(Integer, nullable=True)
    video_views = Column(Integer, nullable=True)
    rank_diff = Column(Integer, nullable=True)
    rank_diff_type = Column(Integer, nullable=True)
    trend = Column(JSON, nullable=True)
    industry_info = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_weekly_trend_hashtag_global_rank", "global_report_id", "rank"),)


class WeeklyTrendSound(Base):
    """Weekly top 100 sounds from TikTok Creative Radar for a given global report."""
    __tablename__ = "weekly_trend_sound"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    global_report_id = Column(Integer, nullable=False, index=True)
    period_start = Column(DateTime, nullable=True)  # 周一 00:00 UTC
    period_end = Column(DateTime, nullable=True)  # 下周一 00:00 UTC
    rank = Column(Integer, nullable=False)
    clip_id = Column(String, nullable=True)
    song_id = Column(String, nullable=True)
    title = Column(String, nullable=True)
    author = Column(String, nullable=True)
    country_code = Column(String, nullable=True)
    duration = Column(Integer, nullable=True)
    link = Column(String, nullable=True)
    trend = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_weekly_trend_sound_global_rank", "global_report_id", "rank"),)


class WeeklyTrendCreator(Base):
    """Weekly top 100 creators from TikTok Creative Radar for a given global report."""
    __tablename__ = "weekly_trend_creator"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    global_report_id = Column(Integer, nullable=False, index=True)
    period_start = Column(DateTime, nullable=True)  # 周一 00:00 UTC
    period_end = Column(DateTime, nullable=True)  # 下周一 00:00 UTC
    rank = Column(Integer, nullable=False)
    tcm_id = Column(String, nullable=True)
    user_id = Column(String, nullable=True, index=True)
    nick_name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    country_code = Column(String, nullable=True)
    follower_cnt = Column(Integer, nullable=True)
    liked_cnt = Column(Integer, nullable=True)
    tt_link = Column(String, nullable=True)
    items = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_weekly_trend_creator_global_rank", "global_report_id", "rank"),)


class OutfitCatalog(Base):
    __tablename__ = "f_outfit_catalog"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    entity_item_treasure_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    internal_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    set_series: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    quality: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    internal_name_accessory: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    internal_name_overlay: Mapped[str] = mapped_column(String(255), nullable=False)
    name_display_text: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    belongs_to_series: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    description_cn: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description_en: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    is_stackable: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    appraised_state: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    """Base model mixin with common timestamp fields"""
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Future: CDN image URL
    pic: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    __table_args__ = (
        Index("ix_f_outfit_catalog_overlay", "internal_name_overlay", unique=True),
    )
