Now the archive redirect endpoint could return 200 `{ "status": "authentication already completed", "username": "<name>", "sec_user_id": "<id>"}`.

so basically it also returns the username and the sec user id on the completed status


  ### A) Email capture (pre-auth)

  - FE → POST /register-email (device headers only, body
    {email})
  - BE → upserts device_emails[device_id] = email
  - If the device already has a finalized auth, BE also
    upserts app_user_emails and do NOT enqueue email_send

  ### B) Start auth

  - FE → POST /link/tiktok/start (device headers)
  - BE → Archive start-auth → returns archive_job_id
  - BE → creates app_auth_jobs row keyed by archive_job_id
    (stores device_id, email snapshot, device metadata)
  - FE stores archive_job_id

  ### C) Poll redirect (this is the frontend loop)

  - FE → GET /link/tiktok/redirect?job_id=<archive_job_id>
    (device headers, time_zone), /link/tiktok/redirect?job_id=...&time_zone=America%2FNew_York
  - BE → Archive get-redirect
      - If pending → returns pending
      - If redirect_url present → returns ready +
        redirect_url and enqueues one worker job
        xordi_finalize:<archive_job_id>
      - If “authentication already completed” → BE does
        not enqueue finalize here; it returns `finalizing`
        unless a session already exists, in which case it
        returns `completed` with `{app_user_id, token}`
  - FE will NO LONGER poll get-authorization-code or call /verify-region or call /finalize

  ### D) Worker finalize (backend-only)

  - Worker job xordi_finalize:
      - polls GET /archive/xordi/get-authorization-code?
        archive_job_id=... until it gets a code (202
        doesn’t count as retries)
      - calls POST /archive/xordi/finalize at most once
      - on success: upserts app_users (canonical
        app_user_id = archive_user_id), mints a session
        row in app_sessions, marks
        app_auth_jobs.status=finalized, then runs the
        region probe and enqueues wrapped pipeline if
        supported
      - on failure: marks auth failed/expired and
        schedules a delayed reauth_notify

  ### E) FE receives app_user_id (+ token)

  - FE keeps polling GET /link/tiktok/redirect
  - As soon as BE can find the session and app_user_id linked with this archive job id, it returns
    status="completed" with {app_user_id, token,
    expires_at}

  ### F) Wrapped pipeline + emails

  - If region probe says yes → enqueue
    watch_history_fetch_2025 → wrapped_analysis → (marks
    app_wrapped_runs.status=ready) → enqueues email_send
  - “Reauth” email is a separate delayed job
    (reauth_notify) 1h after failure if the user didn’t
    successfully auth since. (reauth is enqueued whenever our worker in step D finds out the user's region query or authentication fails)

  ### G) Referral

  - FE → POST /referral body {email} OR {app_user_id} (public, no
    session/device required)
    
  ———

  ## Waitlist (Option A) note

  Option A is implemented: waitlist state is stored on
  `device_emails` so auth-failed users (no `app_user_id`)
  can still opt in using `device+email`.

  Behavior:
  - FE → POST /waitlist (device headers + body {email})
  - BE → upserts device_emails[device_id] = email and sets waitlist_opt_in=true
