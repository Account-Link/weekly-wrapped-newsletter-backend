# 部署指南 / Deployment Guide

本文档介绍如何在与 TikTok Wrapped Backend 共享的服务器上部署 Weekly Wrapped Newsletter Backend。

## 前置要求

- Docker 和 Docker Compose 已安装
- 服务器上已有 TikTok Wrapped Backend 运行（可选，但这是多项目共存的场景）
- PostgreSQL 数据库（可以与 TikTok Wrapped 共用实例，但使用不同的数据库名）
- AWS SES 配置（用于发送邮件）
- Archive API 访问权限

## 关键配置差异

为了在同一服务器上运行两个项目，以下配置必须不同：

| 配置项 | TikTok Wrapped | Weekly Wrapped Newsletter |
|--------|----------------|---------------------------|
| 端口 | 8080 | 8081 |
| 容器名前缀 | `tk-wrapped-*` | `weekly-wrapped-*` |
| Docker Compose 项目名 | `tk-wrapped-backend` | `weekly-wrapped-newsletter-backend` |
| 数据库名 | `tk_wrapped` | `weekly_wrapped_newsletter` |

## 部署步骤

### 1. 克隆项目并配置环境变量

```bash
cd /path/to/your/projects
git clone <repository-url> weekly-wrapped-newsletter-backend
cd weekly-wrapped-newsletter-backend

# 复制环境变量模板
cp .env.example .env
```

### 2. 编辑 `.env` 文件

确保以下关键配置项正确设置：

```bash
# 使用独立的端口（避免与 TikTok Wrapped 的 8080 冲突）
PORT=8081

# 使用独立的数据库（可以在同一 PostgreSQL 实例上）
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/weekly_wrapped_newsletter

# 其他必填项
SECRET_KEY=<生成一个新的密钥: openssl rand -hex 32>
ARCHIVE_BASE_URL=http://localhost:8012
ARCHIVE_API_KEY=<your_archive_api_key>

# AWS SES 配置（可以与 TikTok Wrapped 共用）
AWS_EMAIL="Weekly Wrapped <noreply@example.com>"
AWS_REPLY_TO=team@teleport.computer
AWS_ACCESS_KEY_ID=<your_aws_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret>
AWS_REGION=us-east-1

# S3 配置
S3_BUCKET=feedling
S3_UPLOAD_PREFIX=weekly-uploads
S3_URL=https://diiz1ua008deo.cloudfront.net

# Weekly report 配置
WEEKLY_TOKEN=<generate_a_secure_token>
WEEKLY_REPORT_NODE_URL=<your_node_service_url>
WEEKLY_REPORT_NODE_TOKEN=<your_node_service_token>
```

### 3. 创建数据库（如果使用新数据库）

如果使用独立的数据库，需要先创建：

```bash
# 连接到 PostgreSQL
psql -h <host> -U <user> -d postgres

# 创建新数据库
CREATE DATABASE weekly_wrapped_newsletter;

# 退出
\q
```

### 4. 运行数据库迁移

```bash
sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"
```

### 5. 启动服务

```bash
# 停止旧容器（如果存在）并启动新服务
sudo docker compose down && sudo docker compose up -d --build \
  --scale cron-worker=2 \
  --scale cron-worker-watch=5 \
  --scale cron-worker-auth=2 \
  web cron-worker cron-worker-watch cron-worker-auth
```

### 6. 验证部署

```bash
# 检查容器状态
sudo docker compose ps

# 应该看到以下容器运行中：
# - weekly-wrapped-web
# - weekly-wrapped-scheduler
# - weekly-wrapped-newsletter-backend-cron-worker-1, cron-worker-2
# - weekly-wrapped-newsletter-backend-cron-worker-watch-1 到 5
# - weekly-wrapped-newsletter-backend-cron-worker-auth-1, auth-2

# 检查健康状态
curl http://localhost:8081/healthz

# 查看日志
sudo docker compose logs -f web
sudo docker compose logs -f cron-worker
```

### 7. 配置 Nginx 反向代理（可选）

如果使用 Nginx 作为反向代理，添加配置：

```nginx
# /etc/nginx/sites-available/weekly-wrapped
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
    }
}
```

重新加载 Nginx：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 常见操作

### 查看日志

```bash
# 所有服务
sudo docker compose logs -f

# 特定服务
sudo docker compose logs -f web
sudo docker compose logs -f cron-worker
sudo docker compose logs -f cron-worker-watch
```

### 重启服务

```bash
# 重启所有服务
sudo docker compose restart

# 重启特定服务
sudo docker compose restart web
sudo docker compose restart cron-worker
```

### 停止服务

```bash
sudo docker compose down
```

### 更新代码并重新部署

```bash
git pull origin main
sudo docker compose down
sudo docker compose up -d --build \
  --scale cron-worker=2 \
  --scale cron-worker-watch=5 \
  --scale cron-worker-auth=2 \
  web cron-worker cron-worker-watch cron-worker-auth
```

### 数据库迁移

```bash
# 运行新的迁移
sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"

# 回滚到上一个版本
sudo docker compose run --rm web sh -lc "uv run alembic downgrade -1"

# 查看当前版本
sudo docker compose run --rm web sh -lc "uv run alembic current"
```

### 进入容器调试

```bash
# 进入 web 容器
sudo docker compose exec web sh

# 进入 worker 容器
sudo docker compose exec cron-worker sh
```

## 监控和维护

### 资源监控

```bash
# 查看容器资源使用
sudo docker stats

# 查看特定项目的容器
sudo docker stats $(sudo docker ps --filter "name=weekly-wrapped" -q)
```

### 日志管理

建议配置日志轮转以避免磁盘空间问题：

```bash
# 清理旧日志
sudo docker compose logs --tail=1000 > /dev/null

# 或使用 Docker 日志驱动配置
# 在 docker-compose.yml 中添加：
# logging:
#   driver: "json-file"
#   options:
#     max-size: "10m"
#     max-file: "3"
```

## 故障排查

### 端口冲突

如果 8081 端口已被占用：

```bash
# 检查端口使用
sudo lsof -i :8081

# 修改 .env 中的 PORT 为其他值（如 8082）
# 然后重新部署
```

### 数据库连接失败

```bash
# 测试数据库连接
sudo docker compose run --rm web sh -lc "python -c 'from app.db import SessionLocal; db = SessionLocal(); print(\"DB OK\")'"
```

### 容器无法启动

```bash
# 查看详细错误日志
sudo docker compose logs web

# 检查配置
sudo docker compose config

# 清理并重建
sudo docker compose down -v
sudo docker compose up -d --build
```

## 与 TikTok Wrapped 共存的注意事项

1. **端口隔离**：确保使用不同的端口（8080 vs 8081）
2. **容器命名**：容器名称已经通过 docker-compose.yml 区分
3. **数据库隔离**：使用不同的数据库名
4. **资源分配**：监控服务器资源，根据需要调整 worker 数量
5. **日志存储**：两个项目的日志都存储在各自的 `./log` 目录

## 安全建议

1. 定期更新 SECRET_KEY
2. 不要在代码中硬编码敏感信息
3. 使用 AWS IAM 角色而不是访问密钥（如果在 EC2 上）
4. 定期备份数据库
5. 配置防火墙规则，只开放必要的端口
6. 使用 HTTPS（通过 Nginx 反向代理）

## 备份和恢复

### 数据库备份

```bash
# 备份数据库
pg_dump -h <host> -U <user> -d weekly_wrapped_newsletter > backup_$(date +%Y%m%d).sql

# 恢复数据库
psql -h <host> -U <user> -d weekly_wrapped_newsletter < backup_20260131.sql
```

### 代码备份

```bash
# 使用 Git 标签标记版本
git tag -a v1.0.0 -m "Production release v1.0.0"
git push origin v1.0.0
```

## 联系支持

如有问题，请联系：team@teleport.computer
