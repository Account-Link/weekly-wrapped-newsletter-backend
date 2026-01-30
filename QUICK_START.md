# 快速开始 / Quick Start

本指南帮助你在 5 分钟内启动 Weekly Wrapped Newsletter Backend。

## 前置要求

- Docker 和 Docker Compose
- PostgreSQL 数据库访问权限
- Archive API 密钥

## 快速部署（3 步）

### 1. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件
nano .env
```

**最少必填项**：

```bash
# 端口配置（避免与其他项目冲突）
PORT=8081

# 数据库配置
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/weekly_wrapped_newsletter

# 安全密钥（使用下面命令生成）
SECRET_KEY=$(openssl rand -hex 32)

# Archive API
ARCHIVE_BASE_URL=http://localhost:8012
ARCHIVE_API_KEY=your_archive_api_key
```

### 2. 运行数据库迁移

```bash
sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"
```

### 3. 启动服务

```bash
sudo docker compose up -d --build \
  --scale cron-worker=2 \
  --scale cron-worker-watch=5 \
  --scale cron-worker-auth=2 \
  web cron-worker cron-worker-watch cron-worker-auth
```

## 验证部署

```bash
# 检查健康状态
curl http://localhost:8081/healthz

# 预期输出：{"status":"ok"}

# 查看容器状态
sudo docker compose ps

# 查看日志
sudo docker compose logs -f web
```

## 使用部署脚本（推荐）

我们提供了一个交互式部署脚本：

```bash
# 运行部署脚本
./deploy.sh

# 脚本会引导你完成：
# 1. 环境变量配置检查
# 2. 数据库迁移
# 3. 服务启动
# 4. 日志查看
```

## 常用操作

### 查看日志

```bash
# 所有服务
sudo docker compose logs -f

# 只看 web 服务
sudo docker compose logs -f web

# 只看 worker
sudo docker compose logs -f cron-worker
```

### 重启服务

```bash
sudo docker compose restart
```

### 停止服务

```bash
sudo docker compose down
```

### 进入容器调试

```bash
# 进入 web 容器
sudo docker compose exec web sh

# 在容器内运行 Python
python -c "from app.db import SessionLocal; print('DB OK')"
```

## 测试 API

### 健康检查

```bash
curl http://localhost:8081/healthz
curl http://localhost:8081/readyz
```

### 查看 API 文档

在浏览器中打开：

```
http://localhost:8081/docs
```

这会显示 Swagger UI，包含所有可用的 API 端点。

## 环境变量完整配置

如果需要配置所有选项，参考以下完整配置：

```bash
# === 必填配置 ===

# 服务端口
PORT=8081

# 数据库连接
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/weekly_wrapped_newsletter

# 安全密钥（用于 token 加密）
SECRET_KEY=your_secret_key_here

# Archive API
ARCHIVE_BASE_URL=http://localhost:8012
ARCHIVE_API_KEY=your_archive_api_key

# === AWS 配置 ===

# SES 邮件服务
AWS_EMAIL="Weekly Wrapped <noreply@example.com>"
AWS_REPLY_TO=team@teleport.computer
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1

# S3 存储
S3_BUCKET=your_bucket_name
S3_UPLOAD_PREFIX=weekly-uploads
S3_URL=https://your-cloudfront-url.cloudfront.net

# === 可选配置 ===

# 日志级别
LOG_LEVEL=INFO
LOG_BODY_MAX_CHARS=8000

# Session 配置
SESSION_TTL_DAYS=30

# CORS 配置
CORS_ALLOW_ORIGINS=*

# Worker 配置
WORKER_JOB_CONCURRENCY=10
WORKER_JOB_LEASE_SECONDS=60
WORKER_JOB_HEARTBEAT_SECONDS=30
WORKER_POLL_INTERVAL=1.0

# Watch History 配置
WATCH_HISTORY_SINCE_DATE=2025-01-01
WATCH_HISTORY_PAGE_LIMIT=200
WATCH_HISTORY_MAX_PAGES=3

# Weekly Report 配置
WEEKLY_TOKEN=your_weekly_token
WEEKLY_REPORT_NODE_URL=your_node_service_url
WEEKLY_REPORT_NODE_TOKEN=your_node_token
WEEKLY_REPORT_COVERAGE_GRACE_HOURS=0
WEEKLY_REPORT_FETCH_MAX_DATA_JOBS=8

# Admin 配置
ADMIN_API_KEY=your_admin_key
```

## 与 TikTok Wrapped 共存

如果在同一服务器上已有 TikTok Wrapped Backend：

### 端口分配

```
TikTok Wrapped:              8080
Weekly Wrapped Newsletter:   8081
```

### 数据库隔离

使用不同的数据库名：

```bash
# TikTok Wrapped
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/tk_wrapped

# Weekly Wrapped Newsletter
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/weekly_wrapped_newsletter
```

### 容器命名

两个项目的容器名称已自动区分，不会冲突：

```
TikTok Wrapped:              tk-wrapped-*
Weekly Wrapped Newsletter:   weekly-wrapped-*
```

## 生产环境配置

### Nginx 反向代理

```nginx
server {
    listen 443 ssl http2;
    server_name weekly.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }
}
```

### HTTPS 配置

如果要在容器内启用 HTTPS：

```bash
# .env 文件
HTTPS_ON=true
GUNICORN_CERTFILE=/certbot/live/yourdomain.com/fullchain.pem
GUNICORN_KEYFILE=/certbot/live/yourdomain.com/privkey.pem

# docker-compose.yml 已配置挂载：
# volumes:
#   - /home/ubuntu/certbot:/certbot:ro
```

## 故障排查

### 容器无法启动

```bash
# 查看详细日志
sudo docker compose logs web

# 检查配置
sudo docker compose config

# 完全重建
sudo docker compose down -v
sudo docker compose up -d --build
```

### 端口冲突

```bash
# 检查端口占用
sudo lsof -i :8081

# 如果被占用，修改 .env 中的 PORT
PORT=8082
```

### 数据库连接失败

```bash
# 测试数据库连接
sudo docker compose run --rm web sh -lc "python -c 'from app.db import SessionLocal; db = SessionLocal(); print(\"DB OK\")'"

# 检查 DATABASE_URL 格式
# 正确格式：postgresql+psycopg://user:pass@host:5432/dbname
```

### Worker 不工作

```bash
# 查看 worker 日志
sudo docker compose logs -f cron-worker

# 检查任务队列
sudo docker compose exec web sh
python -c "from app.db import SessionLocal; from app.models import AppJob; db = SessionLocal(); print(db.query(AppJob).count())"
```

## 下一步

- 📖 阅读 [完整部署指南](./DEPLOYMENT.md)
- 🔍 查看 [项目对比文档](./PROJECT_COMPARISON.md)
- 📝 了解 [API 端点](./ENDPOINTS.md)
- 🏗️ 查看 [设计文档](./DESIGN.md)

## 需要帮助？

- 📧 Email: team@teleport.computer
- 📚 文档: 查看项目中的 `docs/` 目录
- 🐛 问题反馈: 创建 GitHub Issue

---

**提示**: 使用 `./deploy.sh` 脚本可以更方便地管理部署！
