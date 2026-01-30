# 项目对比：TikTok Wrapped vs Weekly Wrapped Newsletter

本文档说明 Weekly Wrapped Newsletter Backend 与 TikTok Wrapped Backend 的差异和共同点。

## 相同点

两个项目共享以下技术栈和架构：

- **技术栈**: FastAPI, SQLAlchemy, Alembic, PostgreSQL
- **部署方式**: Docker + Docker Compose
- **Job Queue**: 基于数据库的异步任务队列
- **邮件服务**: AWS SES
- **存储服务**: AWS S3
- **Archive API**: 共用同一个 Archive 服务

## 关键差异

### 1. 端口配置

| 项目 | 端口 | 原因 |
|------|------|------|
| TikTok Wrapped | 8080 | 原始配置 |
| Weekly Wrapped Newsletter | 8081 | 避免端口冲突 |

### 2. Docker 容器命名

| 组件 | TikTok Wrapped | Weekly Wrapped Newsletter |
|------|----------------|---------------------------|
| Web 服务 | `tk-wrapped-web` | `weekly-wrapped-web` |
| Scheduler | `tk-wrapped-backend-cron-scheduler-*` | `weekly-wrapped-scheduler` |
| Worker | `tk-wrapped-backend-cron-worker-*` | `weekly-wrapped-newsletter-backend-cron-worker-*` |

### 3. 数据库

| 项目 | 推荐数据库名 | 说明 |
|------|-------------|------|
| TikTok Wrapped | `tk_wrapped` | 原有数据库 |
| Weekly Wrapped Newsletter | `weekly_wrapped_newsletter` | 独立数据库，避免数据混淆 |

**注意**: 可以使用同一个 PostgreSQL 实例，但强烈建议使用不同的数据库名。

### 4. 项目用途

| 项目 | 用途 | 数据源 |
|------|------|--------|
| TikTok Wrapped | TikTok 年度总结 | 用户 TikTok 观看历史 |
| Weekly Wrapped Newsletter | 周报 Newsletter | 待定（根据实际需求调整） |

### 5. 环境变量差异

```bash
# TikTok Wrapped
PORT=8080
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/tk_wrapped

# Weekly Wrapped Newsletter
PORT=8081
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/weekly_wrapped_newsletter
```

## 服务器共存配置

在同一台服务器上运行两个项目时，需要注意：

### 1. 端口分配

```
TikTok Wrapped:              8080
Weekly Wrapped Newsletter:   8081
```

### 2. 资源管理

监控服务器资源使用，根据需要调整 worker 数量：

```bash
# TikTok Wrapped (假设在 /path/to/tk-wrapped)
cd /path/to/tk-wrapped
sudo docker compose up -d --scale cron-worker=2 --scale cron-worker-watch=3

# Weekly Wrapped Newsletter (在本项目目录)
cd /path/to/weekly-wrapped-newsletter-backend
sudo docker compose up -d --scale cron-worker=2 --scale cron-worker-watch=5
```

### 3. Nginx 配置示例

如果使用 Nginx 作为反向代理：

```nginx
# TikTok Wrapped
server {
    listen 443 ssl http2;
    server_name tee.feedling.app;
    
    location / {
        proxy_pass http://localhost:8080;
        # ... 其他配置
    }
}

# Weekly Wrapped Newsletter
server {
    listen 443 ssl http2;
    server_name weekly.feedling.app;
    
    location / {
        proxy_pass http://localhost:8081;
        # ... 其他配置
    }
}
```

### 4. 日志管理

每个项目都有自己的日志目录：

```
/path/to/tk-wrapped/log/
/path/to/weekly-wrapped-newsletter-backend/log/
```

### 5. 监控命令

```bash
# 查看所有 Docker 容器
sudo docker ps

# 查看 TikTok Wrapped 容器
sudo docker ps --filter "name=tk-wrapped"

# 查看 Weekly Wrapped Newsletter 容器
sudo docker ps --filter "name=weekly-wrapped"

# 资源使用情况
sudo docker stats
```

## 数据库结构对比

两个项目使用相同的数据库 schema（从 Alembic migrations 生成），但数据完全独立：

### 主要表结构

- `app_users`: 用户信息
- `app_auth_jobs`: 认证任务
- `app_jobs`: 异步任务队列
- `watch_history_data`: 观看历史数据
- `device_emails`: 设备邮箱绑定
- `weekly_report`: 周报数据（如果使用）
- `referrals`: 推荐数据

## 部署检查清单

在部署 Weekly Wrapped Newsletter 之前，确保：

- [ ] 已创建独立的 `.env` 文件
- [ ] `PORT` 设置为 8081（或其他未使用的端口）
- [ ] `DATABASE_URL` 指向独立的数据库
- [ ] `SECRET_KEY` 使用新生成的密钥
- [ ] Docker 容器名称不冲突
- [ ] 已运行数据库迁移
- [ ] Nginx 配置（如果使用）正确设置
- [ ] 防火墙规则允许新端口
- [ ] 监控系统已配置（如果有）

## 成本考虑

### 共享资源

两个项目可以共享：
- AWS SES（同一个发送账户）
- S3 Bucket（使用不同的前缀）
- Archive API（如果有配额限制，需要注意）

### 独立资源

需要独立的：
- PostgreSQL 数据库（推荐独立，但可共用实例）
- Docker 容器资源（CPU、内存）

### 服务器资源建议

对于同时运行两个项目：

| 资源 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 2 核 | 4 核 |
| 内存 | 4 GB | 8 GB |
| 磁盘 | 50 GB | 100 GB SSD |
| 带宽 | 100 Mbps | 1 Gbps |

## 故障隔离

两个项目完全独立，一个项目的故障不会影响另一个：

- 独立的 Docker Compose 项目
- 独立的容器
- 独立的数据库
- 独立的日志
- 独立的健康检查

## 迁移策略

如果将来需要合并两个项目或分离到不同服务器：

### 分离到不同服务器

1. 在新服务器上部署 Weekly Wrapped Newsletter
2. 备份数据库并迁移
3. 更新 DNS 或 Nginx 配置
4. 在原服务器上停止 Weekly Wrapped Newsletter

### 合并项目（如果需要）

1. 统一数据库 schema
2. 合并 Docker Compose 配置
3. 统一环境变量管理
4. 更新应用代码以支持多租户

## 总结

Weekly Wrapped Newsletter Backend 是基于 TikTok Wrapped Backend 的独立项目，两者可以在同一服务器上和平共存。关键是确保：

1. **端口不冲突**: 使用不同的端口号
2. **容器隔离**: 使用不同的容器名称
3. **数据隔离**: 使用不同的数据库
4. **资源监控**: 确保服务器资源充足
5. **独立配置**: 各自的 `.env` 文件和配置

通过遵循本文档的指导，可以确保两个项目稳定运行而不会相互干扰。
