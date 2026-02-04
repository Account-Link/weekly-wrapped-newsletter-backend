# Weekly Wrapped Newsletter Backend

Backend for the Weekly Wrapped Newsletter app. Uses FastAPI, SQLAlchemy + Alembic, Postgres (Neon), and a DB-backed job queue. This project is adapted from TikTok Wrapped Backend and can run alongside it on the same server.

## 📚 文档导航

### 🎯 新手入门
- 📖 **[项目概览](./PROJECT_OVERVIEW.md)** - 项目介绍、结构和完整指南
- 🚀 **[快速开始](./QUICK_START.md)** - 5 分钟快速部署指南
- ✅ **[部署检查清单](./DEPLOYMENT_CHECKLIST.md)** - 部署前后检查项

### 📘 部署和配置
- 📦 **[部署指南](./DEPLOYMENT.md)** - 完整的生产环境部署文档
- 🔄 **[项目对比](./PROJECT_COMPARISON.md)** - 与 TikTok Wrapped 项目的差异对比
- 📝 **[迁移总结](./MIGRATION_SUMMARY.md)** - 从 TikTok Wrapped 迁移的所有修改

### 📗 技术文档
- 🏗️ **[设计文档](./DESIGN.md)** - 架构设计和技术细节
- 🔌 **[API 端点](./ENDPOINTS.md)** - API 接口文档

### 🛠️ 配置参考
- 📄 **[Nginx 配置示例](./nginx.conf.example)** - 反向代理配置模板
- ⚙️ **[环境变量模板](./.env.template)** - 详细的环境变量说明
- 🚀 **[部署脚本](./deploy.sh)** - 交互式部署工具
- 📊 **[TikTok Creative Radar](./docs/tiktok-creative-radar.md)** - 周报趋势拉取（数据库手动配置 header + 管理页手动抓取 + 已有数据复用）

> 💡 **首次使用？** 建议按顺序阅读：项目概览 → 快速开始 → 部署检查清单

## 线上域名
[https://tee.feedling.app:8081/](https://tee.feedling.app:8081/)

## ⚡ 快速开始

**最快部署方式**（推荐）：

```bash
# 1. 配置环境变量
cp .env.example .env
nano .env  # 编辑必填项：PORT, DATABASE_URL, SECRET_KEY, ARCHIVE_API_KEY

# 2. 使用部署脚本
./deploy.sh
```

**或者手动部署**：

查看下方的详细步骤。

## 如何运行 / How to Run

### 环境要求

- Python 3.11+
- PostgreSQL（或 Neon 等兼容服务）
- [uv](https://github.com/astral-sh/uv)（推荐）或 pip

### 本地开发（不跑 Docker）

1. **复制环境变量并填写必填项**
   ```sh
   cp .env.example .env
   ```
   编辑 `.env`，至少填写：
   - `DATABASE_URL`：Postgres 连接串，例如 `postgresql+psycopg://user:pass@localhost:5432/dbname`
   - `SECRET_KEY`：用于 token 加密，可用 `openssl rand -hex 32` 生成
   - `ARCHIVE_BASE_URL`、`ARCHIVE_API_KEY`：对接 Archive 服务时必填

2. **安装依赖**
   ```sh
   uv sync
   ```

3. **执行数据库迁移**
   ```sh
   uv run alembic upgrade head
   ```

4. **启动 API 服务**
   ```sh
   uv run uvicorn app.main:app --reload --port 5000
   ```
   默认 API 地址：`http://127.0.0.1:5000`，文档：`http://127.0.0.1:5000/docs`。

5. **（可选）启动 Worker 处理异步任务**
   另开一个终端：
   ```sh
   uv run python -m app.worker
   ```
   不启动 Worker 时，API 仍可访问，但拉取观看记录、分析、发邮件等任务不会执行。

### Docker 方式

```sh
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，确保：
# - DATABASE_URL 使用独立的数据库名 (如 weekly_wrapped_newsletter)
# - 其他必填配置项

# 2. 运行数据库迁移
sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"

# 3. 启动服务
sudo docker compose down && sudo docker compose up -d --build \
  --scale cron-worker=2 \
  --scale cron-worker-watch=5 \
  --scale cron-worker-auth=2 \
  web cron-worker cron-worker-watch cron-worker-auth

# 4. 查看日志
sudo docker compose logs -f cron-worker

# 可选：调整 worker 数量
# sudo docker compose up --build -d --scale cron-worker=4 --scale cron-scheduler=0 

# 完全重建和清理
# sudo docker compose up --build -d --force-recreate --remove-orphans \
#     --scale cron-worker=4 --scale cron-scheduler=0

```

## 🐳 Docker

### 快速部署

```sh
# 1. 确保 .env 已配置
cp .env.example .env
# 编辑 .env，设置 DATABASE_URL 等必填项

# 2. 运行数据库迁移
docker compose run --rm web sh -lc "uv run alembic upgrade head"

# 3. 启动服务
docker compose -p weekly-wrapped-newsletter-backend up --build -d \
  --scale cron-worker=2 \
  web cron-worker
```

> `web` 提供 API 服务；workers 运行 `app.worker` 处理异步任务。

### 服务说明

- **web**: FastAPI 应用，处理 HTTP 请求
- **cron-scheduler**: 调度器（可选）
- **cron-worker**: 通用任务处理（分析、邮件发送等）
- **cron-worker-watch**: 专门处理观看历史拉取
- **cron-worker-auth**: 专门处理认证任务

更多部署选项请查看 [DEPLOYMENT.md](./DEPLOYMENT.md)。

## API (current behavior)
- Link (device-bound): `POST /link/tiktok/start`, `GET /link/tiktok/redirect`, `GET /link/tiktok/queue-status` (redirect mints/returns bearer token once completed).
- Legacy (frontend should not call): `GET /link/tiktok/code`, `POST /link/tiktok/finalize`.
- Post-finalize (not used by frontend anymore): `POST /link/tiktok/verify-region` (bearer + device headers; worker already probes automatically).
- Device-only: `POST /register-email`, `POST /waitlist`.
- Public: `GET /wrapped/{app_user_id}` (pending/ready).
- Referrals: `POST /referral` (create/get code, public), `POST /referral/impression` (public).
- Health: `/healthz`, `/readyz`.
- Device headers required on device-bound calls: `X-Device-Id`, `X-Platform`, `X-App-Version`, `X-OS-Version`. Bearer tokens are returned by `GET /link/tiktok/redirect` when it reaches `status="completed"`.
- Wrapped payload includes computed metrics plus new analysis fields (personality, niches, brainrot, 2026 keyword, etc.), presentation fields (`cat_name`, `analogy_line`, `scroll_time`), and `accessory_set` (head/body/other from items.csv).

## Backend ↔ Frontend Interaction Flow
1) Start TikTok link (device-bound): `POST /link/tiktok/start` → `archive_job_id`.
2) Poll redirect (device-bound): `GET /link/tiktok/redirect?job_id=...&time_zone=...` until:
   - `status="ready"` (returns `redirect_url`), then continue polling
   - `status="completed"` (returns `{app_user_id, token, expires_at}`)
3) Email capture (device-bound, pre- or post-auth): `POST /register-email {email}`.
4) Waitlist (device-bound): `POST /waitlist {email}` works even if auth fails (no `app_user_id` yet).
5) Result: email link opens the frontend as `/wrapped?app_user_id=<app_user_id>`; frontend calls `GET /wrapped/{app_user_id}` (public).

## Jobs
- DB-backed queue `app_jobs` with leases/backoff/idempotency; worker in `app/worker.py` is async and routes tasks.
- Job flow (auto after availability `yes`): `watch_history_fetch_2025` (cursor-walk back to `WATCH_HISTORY_SINCE_DATE` with backoff), `wrapped_analysis` (aggregates metrics + runs LLM prompts for personality/niches/brainrot/keyword/roast), then `email_send` (SES).
- Concurrency knobs: `WORKER_JOB_CONCURRENCY` (jobs/accounts per worker process), `WATCH_HISTORY_MAX_PAGES` (pages per `start` call). Range: `WATCH_HISTORY_SINCE_DATE`/`WATCH_HISTORY_SINCE_MS`.
- Task filter: `WORKER_TASK_ALLOW` (comma-separated task names). Example: `WORKER_TASK_ALLOW=xordi_finalize` to run a dedicated auth worker.

## 📋 重要说明

- `DATABASE_URL` 是必填项（不支持 SQLite）。使用 psycopg3 连接 Postgres/Neon。
- 本项目使用端口 8081，避免与 TikTok Wrapped Backend（8080）冲突。
- 错误处理使用统一的错误信封；Archive 错误映射待实现。
- 配件选择使用 `items.csv`，包含在 wrapped payloads 中。
- 如果与 TikTok Wrapped Backend 共存，请使用不同的数据库名。

## 🛠️ 常用命令

```bash
# 查看服务状态
sudo docker compose ps

# 查看日志
sudo docker compose logs -f web
sudo docker compose logs -f cron-worker

# 重启服务
sudo docker compose restart

# 停止服务
sudo docker compose down

# 进入容器调试
sudo docker compose exec web sh

# 数据库迁移
sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"

# 查看 API 文档
open http://localhost:8081/docs
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

- Email: team@teleport.computer
- 项目维护: Teleport Team
