# 项目迁移总结 / Migration Summary

## 概述

本文档总结了将 TikTok Wrapped Backend 迁移到 Weekly Wrapped Newsletter Backend 所做的所有修改。

## 已完成的修改

### 1. Docker Compose 配置 (`docker-compose.yml`)

#### 修改内容：

- ✅ 项目名称：`tk-wrapped-backend` → `weekly-wrapped-newsletter-backend`
- ✅ Web 容器名：`tk-wrapped-web` → `weekly-wrapped-web`
- ✅ Scheduler 容器名：添加 `container_name: weekly-wrapped-scheduler`
- ✅ 其他 worker 容器会自动使用项目名前缀

#### 影响：

容器命名现在清晰区分两个项目，不会产生冲突。

### 2. 环境变量配置 (`.env.example`)

#### 修改内容：

- ✅ 端口：`PORT=8080` → `PORT=8081`
- ✅ 数据库名示例：`dbname` → `weekly_wrapped_newsletter`
- ✅ 添加注释说明端口选择原因

#### 影响：

默认配置避免与 TikTok Wrapped Backend 的端口冲突。

### 3. 文档更新 (`README.md`)

#### 修改内容：

- ✅ 项目标题和描述更新
- ✅ 添加文档导航链接
- ✅ 线上域名更新（端口从 8080 → 8081）
- ✅ 添加快速开始章节
- ✅ 更新 Docker 部署说明
- ✅ 添加与 TikTok Wrapped 共存说明
- ✅ 添加常用命令参考
- ✅ 改进文档结构和可读性

#### 影响：

文档清晰说明项目用途和部署方式，便于理解和使用。

### 4. 新增文档

#### 4.1 快速开始指南 (`QUICK_START.md`)

- ✅ 5 分钟快速部署指南
- ✅ 最少必填配置项说明
- ✅ 完整环境变量配置参考
- ✅ 常见操作命令
- ✅ 故障排查指南

#### 4.2 完整部署指南 (`DEPLOYMENT.md`)

- ✅ 详细的部署步骤
- ✅ 配置差异说明
- ✅ Nginx 反向代理配置
- ✅ 常见操作和维护
- ✅ 监控和日志管理
- ✅ 故障排查
- ✅ 备份和恢复策略
- ✅ 安全建议

#### 4.3 项目对比文档 (`PROJECT_COMPARISON.md`)

- ✅ 两个项目的相同点
- ✅ 关键差异对比表
- ✅ 服务器共存配置指南
- ✅ 资源管理建议
- ✅ 成本考虑
- ✅ 迁移策略

#### 4.4 部署检查清单 (`DEPLOYMENT_CHECKLIST.md`)

- ✅ 部署前检查项（服务器、代码、配置）
- ✅ 环境变量配置检查
- ✅ 数据库准备检查
- ✅ 网络和防火墙配置
- ✅ 部署步骤验证
- ✅ 部署后检查
- ✅ 持续维护建议

#### 4.5 Nginx 配置示例 (`nginx.conf.example`)

- ✅ HTTP to HTTPS 重定向
- ✅ SSL/TLS 配置
- ✅ 安全头设置
- ✅ 反向代理配置
- ✅ 健康检查端点
- ✅ WebSocket 支持
- ✅ 详细注释说明

#### 4.6 部署脚本 (`deploy.sh`)

- ✅ 交互式部署脚本
- ✅ 环境检查和验证
- ✅ 多种部署选项（首次、更新、迁移等）
- ✅ 日志查看功能
- ✅ 彩色输出和用户友好提示

#### 4.7 项目迁移总结 (`MIGRATION_SUMMARY.md`)

- ✅ 本文档，总结所有修改

## 配置对比表

### 端口配置

| 项目 | 端口 | 状态 |
|------|------|------|
| TikTok Wrapped Backend | 8080 | 保持不变 |
| Weekly Wrapped Newsletter Backend | 8081 | ✅ 已配置 |

### 容器命名

| 组件 | TikTok Wrapped | Weekly Wrapped Newsletter | 状态 |
|------|----------------|---------------------------|------|
| Web | `tk-wrapped-web` | `weekly-wrapped-web` | ✅ 已修改 |
| Scheduler | 默认 | `weekly-wrapped-scheduler` | ✅ 已修改 |
| Worker | `tk-wrapped-backend-*` | `weekly-wrapped-newsletter-backend-*` | ✅ 自动区分 |

### 数据库配置

| 项目 | 推荐数据库名 | 状态 |
|------|-------------|------|
| TikTok Wrapped | `tk_wrapped` | 保持不变 |
| Weekly Wrapped Newsletter | `weekly_wrapped_newsletter` | ✅ 文档已说明 |

## 未修改的部分

以下部分保持与 TikTok Wrapped Backend 相同：

- ✅ 应用代码逻辑（`app/` 目录）
- ✅ 数据库迁移（`alembic/` 目录）
- ✅ 依赖配置（`pyproject.toml`, `uv.lock`）
- ✅ Dockerfile
- ✅ Gunicorn 配置
- ✅ 其他辅助文件

**原因**: 这些是业务逻辑和核心功能，与部署配置无关。如果需要修改功能，应该根据实际业务需求单独调整。

## 部署前准备清单

在部署之前，请确保：

- [ ] ✅ 已阅读 [QUICK_START.md](./QUICK_START.md)
- [ ] ✅ 已阅读 [DEPLOYMENT.md](./DEPLOYMENT.md)
- [ ] ✅ 已阅读 [PROJECT_COMPARISON.md](./PROJECT_COMPARISON.md)
- [ ] ✅ 已完成 [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md) 中的所有检查项
- [ ] ⚠️ 已创建并配置 `.env` 文件（从 `.env.example` 复制）
- [ ] ⚠️ 已确认端口 8081 未被占用
- [ ] ⚠️ 已创建独立的数据库
- [ ] ⚠️ 已配置所有必填的环境变量

## 快速部署命令

```bash
# 1. 克隆或进入项目目录
cd /path/to/weekly-wrapped-newsletter-backend

# 2. 配置环境变量
cp .env.example .env
nano .env  # 编辑配置

# 3. 运行部署脚本
./deploy.sh
# 选择选项 1: 首次部署

# 或者手动部署
sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"
sudo docker compose up -d --build \
  --scale cron-worker=2 \
  --scale cron-worker-watch=5 \
  --scale cron-worker-auth=2 \
  web cron-worker cron-worker-watch cron-worker-auth

# 4. 验证部署
curl http://localhost:8081/healthz
sudo docker compose ps
sudo docker compose logs -f
```

## 与 TikTok Wrapped 共存验证

如果在同一服务器上运行两个项目，验证以下内容：

```bash
# 1. 检查两个项目的容器都在运行
sudo docker ps --filter "name=tk-wrapped"
sudo docker ps --filter "name=weekly-wrapped"

# 2. 检查端口监听
sudo lsof -i :8080  # TikTok Wrapped
sudo lsof -i :8081  # Weekly Wrapped Newsletter

# 3. 测试两个服务的健康检查
curl http://localhost:8080/healthz  # TikTok Wrapped
curl http://localhost:8081/healthz  # Weekly Wrapped Newsletter

# 4. 检查资源使用
sudo docker stats
```

## 文档结构

项目现在包含以下文档：

```
weekly-wrapped-newsletter-backend/
├── README.md                      # 主文档（已更新）
├── QUICK_START.md                 # 快速开始（新增）
├── DEPLOYMENT.md                  # 完整部署指南（新增）
├── DEPLOYMENT_CHECKLIST.md        # 部署检查清单（新增）
├── PROJECT_COMPARISON.md          # 项目对比（新增）
├── MIGRATION_SUMMARY.md           # 本文档（新增）
├── nginx.conf.example             # Nginx 配置示例（新增）
├── deploy.sh                      # 部署脚本（新增）
├── DESIGN.md                      # 设计文档（原有）
├── ENDPOINTS.md                   # API 端点（原有）
├── .env.example                   # 环境变量模板（已更新）
├── docker-compose.yml             # Docker 配置（已更新）
└── ...
```

## 下一步行动

### 对于开发者

1. **本地开发**：
   - 参考 README.md 中的"本地开发"部分
   - 使用不同的端口和数据库

2. **功能开发**：
   - 根据 Weekly Wrapped Newsletter 的实际需求修改业务逻辑
   - 更新 API 端点（如果需要）
   - 添加新的数据模型（如果需要）

3. **测试**：
   - 编写测试用例
   - 测试与 Archive API 的集成
   - 测试邮件发送功能

### 对于运维人员

1. **部署到测试环境**：
   - 使用 [QUICK_START.md](./QUICK_START.md) 快速部署
   - 验证所有功能正常

2. **部署到生产环境**：
   - 完成 [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md) 中的所有检查
   - 使用 [DEPLOYMENT.md](./DEPLOYMENT.md) 作为详细指南
   - 配置监控和告警

3. **持续维护**：
   - 定期检查日志
   - 监控资源使用
   - 定期备份数据库
   - 保持依赖更新

## 注意事项

### ⚠️ 重要提醒

1. **数据隔离**：两个项目必须使用不同的数据库，避免数据混淆
2. **端口冲突**：确保使用不同的端口（8080 vs 8081）
3. **环境变量**：每个项目有独立的 `.env` 文件，不要混用
4. **SECRET_KEY**：每个项目应该使用不同的 SECRET_KEY
5. **容器资源**：监控服务器资源，根据需要调整 worker 数量

### 🔒 安全建议

1. **.env 文件**：
   - 不要提交到 Git
   - 设置正确的文件权限（600 或 400）
   - 定期更新敏感信息

2. **API 文档**：
   - 生产环境考虑限制 `/docs` 和 `/redoc` 访问
   - 使用 Nginx 配置访问控制

3. **SSL/TLS**：
   - 使用 HTTPS（通过 Nginx 或容器内）
   - 定期更新证书

4. **备份**：
   - 定期备份数据库
   - 保存 `.env` 文件的安全副本

## 问题和支持

### 常见问题

**Q: 可以使用相同的数据库实例吗？**
A: 可以，但必须使用不同的数据库名（例如 `tk_wrapped` 和 `weekly_wrapped_newsletter`）。

**Q: 可以共用 AWS 资源吗？**
A: 可以共用 SES 和 S3，但建议使用不同的 S3 前缀。

**Q: 如何回滚到旧版本？**
A: 使用 Git 切换到旧版本，运行数据库回滚（如果需要），然后重新部署。

**Q: 如何监控两个项目的资源使用？**
A: 使用 `docker stats`、系统监控工具（如 Prometheus + Grafana）或云服务商的监控功能。

### 获取帮助

- 📧 Email: team@teleport.computer
- 📚 查看项目文档
- 🐛 GitHub Issues
- 💬 团队内部沟通渠道

## 总结

✅ **完成的工作**：

- 所有配置文件已更新以支持独立部署
- 创建了完整的文档体系
- 提供了部署脚本和检查清单
- 确保与 TikTok Wrapped Backend 可以共存

✅ **可以开始部署**：

项目已经准备好部署到生产环境。按照文档步骤操作即可。

✅ **持续改进**：

根据实际使用情况，可能需要：
- 调整 worker 数量
- 优化性能配置
- 根据业务需求修改功能
- 添加更多监控和告警

---

**祝部署顺利！** 🚀

如有任何问题，请参考相关文档或联系技术支持团队。
