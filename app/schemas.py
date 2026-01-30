from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field
from pydantic import model_validator


class ErrorResponse(BaseModel):
    error: str
    message: Optional[str] = None


class LinkStartResponse(BaseModel):
    archive_job_id: str
    expires_at: Optional[datetime] = None
    queue_position: Optional[int] = None
    failed_count: Optional[int] = None


class RedirectResponse(BaseModel):
    redirect_url: Optional[str] = None
    status: Literal[
        "pending",
        "ready",
        "finalizing",
        "completed",
        "expired",
        "reauth_needed",
        "unsupported_region",
    ] = "pending"
    queue_status: Optional[str] = None
    queue_position: Optional[int] = None
    queue_request_id: Optional[str] = None
    message: Optional[str] = None
    app_user_id: Optional[str] = None
    token: Optional[str] = None
    expires_at: Optional[datetime] = None
    platform_username: Optional[str] = None


class RedirectClickRequest(BaseModel):
    archive_job_id: str


class CodeResponse(BaseModel):
    authorization_code: Optional[str] = None
    status: Literal["pending", "ready", "expired"] = "pending"
    expires_at: Optional[datetime] = None
    queue_position: Optional[int] = None


class FinalizeRequest(BaseModel):
    archive_job_id: str
    authorization_code: str
    time_zone: Optional[str] = None
    referral_code: Optional[str] = None


class FinalizeResponse(BaseModel):
    archive_user_id: str
    sec_user_id: str
    anchor_token: Optional[str] = None
    app_user_id: str
    token: str
    expires_at: datetime
    platform_username: Optional[str] = None


class VerifyRegionResponse(BaseModel):
    is_watch_history_available: Literal["unknown", "yes", "no"]
    attempts: int
    last_error: Optional[str] = None


class RegisterEmailRequest(BaseModel):
    email: EmailStr
    referral_code: Optional[str] = None


class WaitlistRequest(BaseModel):
    email: EmailStr


class DataJobRef(BaseModel):
    id: str
    status: Literal["pending", "running", "succeeded", "failed", "unknown"] = "pending"


class AccessoryItem(BaseModel):
    internal_name: str
    reason: str


class AccessorySet(BaseModel):
    head: AccessoryItem
    body: AccessoryItem
    other: AccessoryItem


class ScrollTime(BaseModel):
    title: str
    rate: str
    between_time: str


class WrappedPayload(BaseModel):
    total_hours: float
    total_videos: int
    night_pct: float
    peak_hour: Optional[int] = None
    top_music: Dict[str, Any]
    top_creators: List[str]
    cat_name: str
    analogy_line: Optional[str] = None
    scroll_time: ScrollTime
    personality_type: str
    personality_explanation: Optional[str] = None
    niche_journey: List[str]
    top_niches: List[str]
    top_niche_percentile: Optional[str] = None
    brain_rot_score: int
    brainrot_intensity: Optional[int] = None
    brainrot_volume_hours: Optional[float] = None
    brainrot_confidence: Optional[float] = None
    brainrot_enriched_hours: Optional[float] = None
    brainrot_enriched_watch_pct: Optional[float] = None
    brain_rot_explanation: Optional[str] = None
    keyword_2026: str
    thumb_roast: Optional[str] = None
    platform_username: Optional[str] = None
    email: Optional[str] = None
    source_spans: List[Dict[str, Any]] = []
    data_jobs: Dict[str, DataJobRef]
    accessory_set: AccessorySet


class WrappedEnqueueResponse(BaseModel):
    status: Literal["pending", "ready"]
    wrapped_run_id: Optional[str] = None
    existing_run_id: Optional[str] = None
    email_delivery: Optional[str] = None
    wrapped: Optional[WrappedPayload] = None
    queue_position: Optional[int] = None
    queue_eta_seconds: Optional[int] = None
    queue_status: Optional[str] = None


class ReferralResponse(BaseModel):
    code: str
    referral_url: Optional[str] = None


class ReferralRequest(BaseModel):
    app_user_id: Optional[str] = None
    email: Optional[EmailStr] = None

    @model_validator(mode="after")
    def _require_identity(self) -> "ReferralRequest":
        if not (self.app_user_id or self.email):
            raise ValueError("app_user_id_or_email_required")
        return self


class ReferralImpressionRequest(BaseModel):
    code: str


class WrappedStatusResponse(BaseModel):
    status: Literal["pending", "ready"]
    wrapped_run_id: str
    wrapped: Optional[WrappedPayload] = None
    watch_history_progress: Optional[Dict[str, Any]] = None
    queue_position: Optional[int] = None
    queue_eta_seconds: Optional[int] = None
    queue_status: Optional[str] = None


class AdminTestRunRequest(BaseModel):
    sec_user_id: str
    email: EmailStr
    platform_username: Optional[str] = None
    time_zone: Optional[str] = "UTC"
    app_user_id: Optional[str] = None
    force_new_run: bool = True
    scrape_max_videos: Optional[int] = None


class AdminTestRunResponse(BaseModel):
    app_user_id: str
    wrapped_run_id: str
    watch_history_job_id: str
    wrapped_link: str
    wrapped_status_endpoint: str
    admin_status_endpoint: str


class AdminTestRunStatusResponse(BaseModel):
    wrapped_run_id: str
    app_user_id: str
    sec_user_id: str
    run_status: str
    watch_history_job_id: Optional[str] = None
    platform_username: Optional[str] = None
    email: Optional[str] = None
    wrapped_link: str
    wrapped_status_endpoint: str
    data_jobs: Dict[str, Any] = {}
    watch_history_progress: Optional[Dict[str, Any]] = None
    analysis_warnings: Optional[List[str]] = None
    analysis_debug: Optional[Dict[str, Any]] = None
    jobs: Dict[str, Any] = {}


class AdminTestEnqueueResponse(BaseModel):
    wrapped_run_id: str
    task_name: str
    job_id: str
    status: str
    idempotency_key: Optional[str] = None


class AdminVerifyRegionEnqueueResponse(BaseModel):
    app_user_id: str
    task_name: str
    job_id: str
    status: str
    idempotency_key: Optional[str] = None


class AdminUserStageVerify(BaseModel):
    is_watch_history_available: Literal["unknown", "yes", "no"]
    job_id: Optional[str] = None
    job_status: Optional[str] = None
    is_stuck: Optional[bool] = None
    stuck_reason: Optional[str] = None
    locked_for_seconds: Optional[float] = None
    locked_by: Optional[str] = None
    locked_at: Optional[str] = None
    updated_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class AdminUserStageResponse(BaseModel):
    app_user_id: str
    sec_user_id: Optional[str] = None
    wrapped_run_id: Optional[str] = None
    run_status: Optional[str] = None
    stage: str
    next_stage: str
    verify: AdminUserStageVerify
    data_jobs: Dict[str, Any] = {}
    watch_history_progress: Optional[Dict[str, Any]] = None
    jobs: Dict[str, Any] = {}


class AdminUserRestartRequest(BaseModel):
    stage: Optional[str] = None  # next|verify_region|watch_history|analysis|email
    wrapped_run_id: Optional[str] = None
    force_new_jobs: bool = True
    reset_payload: bool = True


class AdminUserRestartResponse(BaseModel):
    app_user_id: str
    wrapped_run_id: Optional[str] = None
    selected_stage: str
    enqueued_task: Optional[str] = None
    job_id: Optional[str] = None
    status: Optional[str] = None
    idempotency_key: Optional[str] = None
    skipped_reason: Optional[str] = None


class AdminUserStageBatchRequest(BaseModel):
    limit: int = 50
    offset: int = 0
    include_admin_test: bool = False
    watch_history_status: Optional[List[Literal["unknown", "yes", "no"]]] = None
    run_status: Optional[List[Literal["pending", "ready", "failed", "no_run"]]] = None
    stage: Optional[List[Literal["verify_region", "watch_history", "analysis", "email", "ready", "no_run"]]] = None


class AdminUserStageBatchResponse(BaseModel):
    limit: int
    offset: int
    requested: int
    returned: int
    items: List[AdminUserStageResponse]


class AdminUserRestartBatchRequest(BaseModel):
    limit: int = 50
    offset: int = 0
    include_admin_test: bool = False
    watch_history_status: Optional[List[Literal["unknown", "yes", "no"]]] = None
    run_status: Optional[List[Literal["pending", "ready", "failed", "no_run"]]] = None
    stage_filter: Optional[List[Literal["verify_region", "watch_history", "analysis", "email", "ready", "no_run"]]] = None
    restart_stage: Optional[Literal["next", "verify_region", "watch_history", "analysis", "email"]] = "next"
    force_new_jobs: bool = True
    reset_payload: bool = True
    dry_run: bool = False


class AdminUserRestartBatchItem(BaseModel):
    app_user_id: str
    result: Optional[AdminUserRestartResponse] = None
    error: Optional[str] = None


class AdminUserRestartBatchResponse(BaseModel):
    limit: int
    offset: int
    requested: int
    processed: int
    enqueued: int
    skipped: int
    dry_run: bool
    results: List[AdminUserRestartBatchItem]


class AdminVerifyRegionBatchRequest(BaseModel):
    include_unknown: bool = False
    auto_enqueue: bool = False
    force_new: bool = False


class AdminVerifyRegionBatchItem(BaseModel):
    app_user_id: str
    sec_user_id: Optional[str] = None
    job_id: Optional[str] = None
    status: Optional[str] = None
    idempotency_key: Optional[str] = None
    error: Optional[str] = None


class AdminVerifyRegionBatchResponse(BaseModel):
    batch_id: str
    matched: int
    processed: int
    enqueued: int
    results: List[AdminVerifyRegionBatchItem]


class AdminVerifyRegionBatchStatusItem(BaseModel):
    app_user_id: Optional[str] = None
    sec_user_id: Optional[str] = None
    job_id: str
    job_status: Optional[str] = None
    verify_status: Optional[str] = None
    attempts: Optional[int] = None
    last_error: Optional[str] = None
    checked_at: Optional[str] = None
    error: Optional[str] = None


class AdminVerifyRegionBatchStatusResponse(BaseModel):
    batch_id: str
    created_at: Optional[str] = None
    total: int
    completed: int
    pending: int
    yes: int
    no: int
    unknown: int
    error: int
    results: List[AdminVerifyRegionBatchStatusItem]


class AdminJobStatusResponse(BaseModel):
    job_id: str
    task_name: str
    status: str
    attempts: int
    max_attempts: int
    not_before: Optional[str] = None
    locked_by: Optional[str] = None
    locked_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    idempotency_key: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None


PromptKey = Literal[
    "personality",
    "personality_explanation",
    "niche_journey",
    "top_niches",
    "brainrot_score",
    "brainrot_explanation",
    "keyword_2026",
    "thumb_roast",
]


class AdminPromptPipelineRequest(BaseModel):
    sec_user_id: str
    time_zone: Optional[str] = None
    prompt_overrides: Dict[PromptKey, str] = Field(default_factory=dict)
    model: Optional[str] = None
    temperature: float = 0.7
    dry_run: bool = False


class AdminPromptCallResult(BaseModel):
    prompt_key: PromptKey
    prompt_used: str
    output_text: Optional[str] = None
    parsed: Optional[Any] = None
    error: Optional[str] = None


class AdminPromptPipelineResponse(BaseModel):
    sec_user_id: str
    model: str
    range: Dict[str, Any]
    sample_texts_used: List[str]
    results: List[AdminPromptCallResult]


class AdminBrainrotScoreRequest(BaseModel):
    sec_user_id: str
    time_zone: Optional[str] = "UTC"
    range: Optional[Dict[str, Any]] = None


class AdminBrainrotScoreResponse(BaseModel):
    sec_user_id: str
    range: Dict[str, Any]
    time_zone: Optional[str] = None
    updated_wrapped_run_id: Optional[str] = None
    brain_rot_score: int
    brainrot_intensity: int
    brainrot_intensity_raw: float
    brainrot_intensity_raw_effective: float
    brainrot_intensity_linear: float
    brainrot_confidence: float
    brainrot_enriched_watch_pct: float
    brainrot_enriched_hours: float
    brainrot_volume_hours: float
    brainrot_quality_weight: float
    brainrot_normalization: Dict[str, Any]


class AdminWrappedRetryRequest(BaseModel):
    """Admin-only: restart an existing wrapped run for a real user."""

    include_watch_history: bool = True
    include_analysis: bool = True
    force_new_jobs: bool = True
    reset_payload: bool = True


class AdminRetryFailedRunsRequest(BaseModel):
    """Admin-only: bulk retry failed wrapped runs."""

    limit: int = 50
    dry_run: bool = False
    reset_payload: bool = True
    # If omitted, backend decides per-run based on which stage failed.
    include_watch_history: Optional[bool] = None
    include_analysis: Optional[bool] = None
    include_email: Optional[bool] = None


class AdminRetryFailedRunsItem(BaseModel):
    wrapped_run_id: str
    app_user_id: Optional[str] = None
    sec_user_id: Optional[str] = None
    action: str
    enqueued_task: Optional[str] = None
    job_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    skipped_reason: Optional[str] = None


class AdminRetryFailedRunsResponse(BaseModel):
    matched: int
    processed: int
    enqueued: int
    dry_run: bool
    results: List[AdminRetryFailedRunsItem]


class AdminRetryZeroVideosRequest(BaseModel):
    limit: Optional[int] = 50
    dry_run: bool = False
    reset_payload: bool = True
    include_admin_test: bool = False
    run_status: Optional[List[Literal["pending", "ready", "failed"]]] = None


class AdminRetryZeroVideosItem(BaseModel):
    wrapped_run_id: str
    app_user_id: Optional[str] = None
    sec_user_id: Optional[str] = None
    total_videos: Optional[int] = None
    watch_history_progress: Optional[Dict[str, Any]] = None
    action: str
    enqueued_task: Optional[str] = None
    job_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    skipped_reason: Optional[str] = None


class AdminRetryZeroVideosResponse(BaseModel):
    matched: int
    processed: int
    enqueued: int
    dry_run: bool
    results: List[AdminRetryZeroVideosItem]


class TopicItem(BaseModel):
    topic: str
    pic_url: str


class WeeklyReportResponse(BaseModel):
    id: int
    app_user_id: str
    email_content: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    send_status: Literal["pending", "sent", "failed"]
    feeding_state: Optional[str] = None
    trend_name: Optional[str] = None
    trend_type: Optional[str] = None
    discovery_rank: Optional[int] = None
    total_discoverers: Optional[int] = None
    origin_niche_text: Optional[str] = None
    spread_end_text: Optional[str] = None
    reach_start: Optional[float] = None
    reach_end: Optional[float] = None
    current_reach: Optional[float] = None
    total_videos: Optional[int] = None
    total_time: Optional[int] = None
    pre_total_time: Optional[int] = None
    miles_scrolled: Optional[int] = None
    topics: Optional[List[TopicItem]] = None
    timezone: Optional[str] = None
    rabbit_hole_datetime: Optional[datetime] = None
    rabbit_hole_date: Optional[str] = None
    rabbit_hole_time: Optional[str] = None
    rabbit_hole_count: Optional[int] = None
    rabbit_hole_category: Optional[str] = None
    nudge_text: Optional[str] = None


class UnsubscribeRequest(BaseModel):
    app_user_id: str


class UnsubscribeResponse(BaseModel):
    success: bool
    message: str


class UploadResponse(BaseModel):
    url: str
    key: Optional[str] = None


class WeeklyReportTestRequest(BaseModel):
    send_email: bool = False  # Whether to send the email after generating the report (default: False for safety)
    email: Optional[EmailStr] = None  # Override email address (for testing, instead of user's email)


class WeeklyReportTestResponse(BaseModel):
    app_user_id: str
    report_id: int
    analyze_job_id: str
    send_job_id: Optional[str] = None
    email_sent: bool
    email_address: Optional[str] = None
    report: WeeklyReportResponse
