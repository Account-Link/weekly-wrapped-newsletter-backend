# 部署检查清单 / Deployment Checklist

在部署 Weekly Wrapped Newsletter Backend 之前，请完成以下检查清单。

## 📋 部署前检查

### 1. 服务器准备

- [ ] 服务器操作系统：Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- [ ] Docker 已安装（版本 20.10+）
  ```bash
  docker --version
  ```
- [ ] Docker Compose 已安装（版本 2.0+）
  ```bash
  docker compose version
  ```
- [ ] 有足够的磁盘空间（推荐至少 50GB 可用）
  ```bash
  df -h
  ```
- [ ] 有足够的内存（推荐至少 4GB）
  ```bash
  free -h
  ```
- [ ] 端口 8081 未被占用
  ```bash
  sudo lsof -i :8081
  ```

### 2. 代码和配置

- [ ] 代码已克隆到服务器
  ```bash
  cd /path/to/weekly-wrapped-newsletter-backend
  git status
  ```
- [ ] `.env` 文件已创建（从 `.env.example` 复制）
  ```bash
  cp .env.example .env
  ```
- [ ] `.env` 文件权限正确（避免敏感信息泄露）
  ```bash
  chmod 600 .env
  ```
- [ ] `.env` 文件已配置所有必填项（见下方）

### 3. 环境变量配置

#### 必填项

- [ ] `PORT=8081`（或其他未被占用的端口）
- [ ] `DATABASE_URL` 已设置并测试连接
  ```bash
  # 测试数据库连接
  psql "$DATABASE_URL" -c "SELECT 1"
  ```
- [ ] `SECRET_KEY` 已生成（至少 32 字节）
  ```bash
  openssl rand -hex 32
  ```
- [ ] `ARCHIVE_BASE_URL` 已设置
- [ ] `ARCHIVE_API_KEY` 已设置

#### AWS 配置（如果需要邮件功能）

- [ ] `AWS_EMAIL` 已配置
- [ ] `AWS_REPLY_TO` 已配置
- [ ] `AWS_ACCESS_KEY_ID` 已配置
- [ ] `AWS_SECRET_ACCESS_KEY` 已配置
- [ ] `AWS_REGION` 已配置
- [ ] AWS SES 已验证发送邮箱地址

#### S3 配置（如果需要文件上传）

- [ ] `S3_BUCKET` 已配置
- [ ] `S3_UPLOAD_PREFIX` 已配置
- [ ] `S3_URL` 已配置（CloudFront URL 或 S3 URL）
- [ ] S3 Bucket 权限已正确配置

#### Weekly Report 配置（如果使用）

- [ ] `WEEKLY_TOKEN` 已生成
- [ ] `WEEKLY_REPORT_NODE_URL` 已配置
- [ ] `WEEKLY_REPORT_NODE_TOKEN` 已配置

### 4. 数据库准备

- [ ] PostgreSQL 服务运行正常
  ```bash
  psql -h <host> -U <user> -c "SELECT version()"
  ```
- [ ] 数据库已创建（推荐名称：`weekly_wrapped_newsletter`）
  ```sql
  CREATE DATABASE weekly_wrapped_newsletter;
  ```
- [ ] 数据库用户有足够权限（CREATE, DROP, ALTER 等）
- [ ] 数据库可以从服务器访问（检查防火墙规则）

### 5. 网络和防火墙

- [ ] 防火墙允许端口 8081（或配置的端口）
  ```bash
  # Ubuntu/Debian with ufw
  sudo ufw allow 8081/tcp
  sudo ufw status
  
  # CentOS/RHEL with firewalld
  sudo firewall-cmd --permanent --add-port=8081/tcp
  sudo firewall-cmd --reload
  ```
- [ ] 如果使用 Nginx，配置已准备就绪
- [ ] SSL 证书已准备（如果使用 HTTPS）
- [ ] DNS 记录已指向服务器

### 6. 与 TikTok Wrapped 共存检查（如果适用）

- [ ] 端口不冲突（TikTok: 8080, Weekly: 8081）
- [ ] 容器名称不冲突（已通过 docker-compose.yml 区分）
- [ ] 数据库名称不同
- [ ] 足够的服务器资源运行两个项目
  ```bash
  # 检查当前资源使用
  sudo docker stats
  ```

## 🚀 部署步骤

### 1. 运行数据库迁移

- [ ] 迁移命令执行成功
  ```bash
  sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"
  ```
- [ ] 检查数据库表已创建
  ```bash
  psql "$DATABASE_URL" -c "\dt"
  ```

### 2. 启动服务

- [ ] 服务启动成功
  ```bash
  sudo docker compose up -d --build \
    --scale cron-worker=2 \
    --scale cron-worker-watch=5 \
    --scale cron-worker-auth=2 \
    web cron-worker cron-worker-watch cron-worker-auth
  ```
- [ ] 所有容器都在运行
  ```bash
  sudo docker compose ps
  ```

### 3. 验证部署

- [ ] 健康检查通过
  ```bash
  curl http://localhost:8081/healthz
  # 预期输出: {"status":"ok"}
  ```
- [ ] Readiness 检查通过
  ```bash
  curl http://localhost:8081/readyz
  ```
- [ ] API 文档可访问
  ```bash
  curl http://localhost:8081/docs
  ```
- [ ] 无错误日志
  ```bash
  sudo docker compose logs --tail=50
  ```

### 4. 功能测试

- [ ] 可以创建测试用户
- [ ] Worker 正常处理任务
  ```bash
  sudo docker compose logs -f cron-worker
  ```
- [ ] 邮件发送功能正常（如果配置）
- [ ] 文件上传功能正常（如果配置）

## 📊 部署后检查

### 1. 监控设置

- [ ] 日志正常输出
  ```bash
  sudo docker compose logs -f
  ```
- [ ] 日志轮转已配置（避免磁盘满）
- [ ] 资源使用在合理范围内
  ```bash
  sudo docker stats
  ```
- [ ] 设置告警（可选）

### 2. 备份策略

- [ ] 数据库自动备份已配置
  ```bash
  # 示例：每天备份
  0 2 * * * pg_dump "$DATABASE_URL" > /backup/weekly-wrapped-$(date +\%Y\%m\%d).sql
  ```
- [ ] `.env` 文件已备份到安全位置
- [ ] 配置文件已版本控制（不包含敏感信息）

### 3. 安全检查

- [ ] `.env` 文件不在 Git 仓库中
  ```bash
  git status --ignored
  ```
- [ ] 敏感文件权限正确
  ```bash
  ls -la .env
  # 应该是 600 或 400
  ```
- [ ] 防火墙规则正确配置
- [ ] SSL/TLS 配置正确（如果使用）
- [ ] API 文档在生产环境中受保护（可选）

### 4. 性能优化

- [ ] Worker 数量根据负载调整
- [ ] 数据库连接池配置合理
- [ ] Gunicorn workers 数量合适
  ```bash
  # .env 中配置
  GUNICORN_WORKERS=4
  ```
- [ ] 日志级别适当（生产环境建议 INFO 或 WARNING）

## 🔄 持续维护

### 定期检查

- [ ] 每周检查日志是否有错误
  ```bash
  sudo docker compose logs --since 7d | grep -i error
  ```
- [ ] 每月检查磁盘空间
  ```bash
  df -h
  du -sh ./log
  ```
- [ ] 定期更新依赖和安全补丁
- [ ] 定期测试备份恢复流程

### 更新流程

部署新版本时的检查清单：

- [ ] 代码已拉取最新版本
  ```bash
  git pull origin main
  ```
- [ ] 查看是否有新的迁移
  ```bash
  git log --oneline alembic/versions/
  ```
- [ ] 运行数据库迁移（如果有）
  ```bash
  sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"
  ```
- [ ] 重新构建并重启服务
  ```bash
  sudo docker compose down
  sudo docker compose up -d --build
  ```
- [ ] 验证服务正常运行
  ```bash
  curl http://localhost:8081/healthz
  sudo docker compose logs --tail=50
  ```

## 🆘 故障排查

如果遇到问题，按以下顺序检查：

1. **查看日志**
   ```bash
   sudo docker compose logs --tail=100
   ```

2. **检查容器状态**
   ```bash
   sudo docker compose ps
   ```

3. **测试数据库连接**
   ```bash
   sudo docker compose exec web sh -c "python -c 'from app.db import SessionLocal; SessionLocal()'"
   ```

4. **检查端口占用**
   ```bash
   sudo lsof -i :8081
   ```

5. **检查资源使用**
   ```bash
   sudo docker stats
   free -h
   df -h
   ```

6. **查看完整配置**
   ```bash
   sudo docker compose config
   ```

## 📞 需要帮助？

如果完成检查清单后仍有问题：

- 📧 Email: team@teleport.computer
- 📚 查看 [DEPLOYMENT.md](./DEPLOYMENT.md) 获取详细信息
- 🐛 查看 [故障排查部分](./DEPLOYMENT.md#故障排查)
- 💬 查看项目 Issues

---

**提示**: 建议打印此清单或保存副本，在每次部署时使用。
