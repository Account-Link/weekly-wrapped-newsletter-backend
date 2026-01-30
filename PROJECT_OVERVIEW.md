# Weekly Wrapped Newsletter Backend - 项目概览

## 🎯 项目简介

Weekly Wrapped Newsletter Backend 是一个基于 FastAPI 的后端服务，用于生成和发送周报 Newsletter。本项目从 TikTok Wrapped Backend 迁移而来，并已配置为可以与原项目在同一服务器上共存。

## 📊 项目状态

| 项目 | 状态 | 说明 |
|------|------|------|
| 配置迁移 | ✅ 完成 | 所有配置文件已更新 |
| 文档完善 | ✅ 完成 | 完整的部署和使用文档 |
| 部署工具 | ✅ 完成 | 提供交互式部署脚本 |
| 测试就绪 | ✅ 就绪 | 可以开始部署和测试 |
| 业务逻辑 | ⚠️ 待定 | 根据实际需求调整 |

## 🏗️ 技术栈

- **框架**: FastAPI (Python 3.11+)
- **数据库**: PostgreSQL with SQLAlchemy
- **迁移**: Alembic
- **任务队列**: 基于数据库的异步任务系统
- **部署**: Docker + Docker Compose
- **邮件**: AWS SES
- **存储**: AWS S3
- **Web 服务器**: Gunicorn + Uvicorn
- **包管理**: uv

## 📁 项目结构

```
weekly-wrapped-newsletter-backend/
│
├── 📚 文档
│   ├── README.md                      # 主要文档
│   ├── QUICK_START.md                 # 快速开始指南（5分钟部署）
│   ├── DEPLOYMENT.md                  # 完整部署指南
│   ├── DEPLOYMENT_CHECKLIST.md        # 部署检查清单
│   ├── PROJECT_COMPARISON.md          # 与 TikTok Wrapped 对比
│   ├── MIGRATION_SUMMARY.md           # 迁移修改总结
│   ├── PROJECT_OVERVIEW.md            # 本文档
│   ├── DESIGN.md                      # 架构设计文档
│   └── ENDPOINTS.md                   # API 端点文档
│
├── 🛠️ 配置和工具
│   ├── docker-compose.yml             # Docker Compose 配置 (✅ 已更新)
│   ├── Dockerfile                     # Docker 镜像配置
│   ├── .env.example                   # 环境变量示例 (✅ 已更新)
│   ├── .env.template                  # 详细环境变量模板 (✅ 新增)
│   ├── nginx.conf.example             # Nginx 配置示例 (✅ 新增)
│   ├── deploy.sh                      # 部署脚本 (✅ 新增)
│   ├── gunicorn_conf.py               # Gunicorn 配置
│   ├── alembic.ini                    # Alembic 配置
│   └── pyproject.toml                 # Python 依赖配置
│
├── 🔧 应用代码
│   ├── app/
│   │   ├── main.py                    # FastAPI 应用入口
│   │   ├── settings.py                # 配置管理
│   │   ├── db.py                      # 数据库连接
│   │   ├── models.py                  # 数据模型
│   │   ├── schemas.py                 # Pydantic schemas
│   │   ├── worker.py                  # 异步任务 worker
│   │   ├── emailer.py                 # 邮件发送
│   │   ├── crypto.py                  # 加密工具
│   │   ├── errors.py                  # 错误处理
│   │   ├── observability.py           # 监控和日志
│   │   ├── routers/                   # API 路由
│   │   └── services/                  # 业务逻辑服务
│   │
│   └── alembic/                       # 数据库迁移
│       └── versions/                  # 迁移脚本
│
└── 📄 其他
    ├── .gitignore                     # Git 忽略文件
    ├── items.csv                      # 配件数据
    └── docs/                          # 额外文档
```

## 🚀 快速开始

### 最快部署（3 步）

```bash
# 1. 配置环境变量
cp .env.example .env
nano .env  # 编辑必填项

# 2. 使用部署脚本
./deploy.sh

# 3. 验证部署
curl http://localhost:8081/healthz
```

### 手动部署

```bash
# 1. 运行数据库迁移
sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"

# 2. 启动服务
sudo docker compose up -d --build \
  --scale cron-worker=2 \
  --scale cron-worker-watch=5 \
  --scale cron-worker-auth=2 \
  web cron-worker cron-worker-watch cron-worker-auth

# 3. 查看日志
sudo docker compose logs -f
```

详细步骤请查看 [QUICK_START.md](./QUICK_START.md)。

## 📖 文档指南

根据你的角色和需求，选择合适的文档：

### 👨‍💻 开发者

1. **首次接触项目**
   - 阅读本文档 (PROJECT_OVERVIEW.md)
   - 查看 [DESIGN.md](./DESIGN.md) 了解架构
   - 查看 [ENDPOINTS.md](./ENDPOINTS.md) 了解 API

2. **本地开发**
   - 查看 [README.md](./README.md) 的本地开发部分
   - 配置 `.env` 文件
   - 运行本地服务

3. **理解迁移**
   - 阅读 [MIGRATION_SUMMARY.md](./MIGRATION_SUMMARY.md)
   - 查看 [PROJECT_COMPARISON.md](./PROJECT_COMPARISON.md)

### 👨‍🔧 运维人员

1. **首次部署**
   - 阅读 [QUICK_START.md](./QUICK_START.md)
   - 完成 [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)
   - 参考 [DEPLOYMENT.md](./DEPLOYMENT.md) 详细步骤

2. **配置和调优**
   - 查看 [.env.template](./.env.template) 了解所有配置项
   - 使用 [nginx.conf.example](./nginx.conf.example) 配置反向代理
   - 根据负载调整 worker 数量

3. **多项目共存**
   - 阅读 [PROJECT_COMPARISON.md](./PROJECT_COMPARISON.md)
   - 确保端口、容器名、数据库不冲突
   - 监控服务器资源使用

### 👨‍💼 项目经理

1. **项目概况**
   - 阅读本文档 (PROJECT_OVERVIEW.md)
   - 查看 [MIGRATION_SUMMARY.md](./MIGRATION_SUMMARY.md) 了解完成的工作

2. **部署状态**
   - 使用 [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md) 跟踪进度

3. **资源规划**
   - 查看 [PROJECT_COMPARISON.md](./PROJECT_COMPARISON.md) 的成本考虑部分
   - 了解服务器资源需求

## 🔑 关键配置

### 端口和网络

| 服务 | 端口 | 说明 |
|------|------|------|
| TikTok Wrapped Backend | 8080 | 原有服务 |
| **Weekly Wrapped Newsletter** | **8081** | 本项目（避免冲突） |
| PostgreSQL | 5432 | 数据库（可共用实例） |
| Nginx | 80/443 | 反向代理（可选） |

### 容器命名

| 组件 | 容器名 | 说明 |
|------|--------|------|
| Web 服务 | `weekly-wrapped-web` | API 服务器 |
| Scheduler | `weekly-wrapped-scheduler` | 任务调度器 |
| Workers | `weekly-wrapped-newsletter-backend-cron-worker-*` | 任务处理器 |

### 数据库

- **推荐数据库名**: `weekly_wrapped_newsletter`
- **与 TikTok Wrapped 隔离**: 使用不同的数据库名
- **可以共用**: PostgreSQL 实例（同一服务器）

## 🛠️ 核心功能

### 当前功能（从 TikTok Wrapped 继承）

- ✅ 用户认证和授权
- ✅ TikTok 账号关联
- ✅ 观看历史获取
- ✅ 数据分析和聚合
- ✅ 邮件发送（AWS SES）
- ✅ 文件上传（AWS S3）
- ✅ 异步任务队列
- ✅ 健康检查端点
- ✅ API 文档（Swagger UI）

### 需要调整的功能（根据实际需求）

- ⚠️ 业务逻辑（周报内容生成）
- ⚠️ 数据模型（如果有新的数据结构）
- ⚠️ API 端点（如果有新的接口需求）
- ⚠️ 定时任务（周报发送时间等）

## 🔐 安全考虑

### 已配置的安全措施

- ✅ `.env` 文件已在 `.gitignore` 中
- ✅ 密钥加密（SECRET_KEY）
- ✅ CORS 配置
- ✅ 健康检查端点
- ✅ 容器隔离

### 建议的额外措施

- 🔒 使用 HTTPS（Nginx 配置示例已提供）
- 🔒 限制 API 文档访问（生产环境）
- 🔒 定期更新密钥和凭证
- 🔒 配置防火墙规则
- 🔒 使用 AWS IAM 角色（而不是访问密钥）
- 🔒 启用数据库备份
- 🔒 配置日志审计

## 📊 监控和维护

### 日志

```bash
# 查看所有日志
sudo docker compose logs -f

# 查看特定服务
sudo docker compose logs -f web
sudo docker compose logs -f cron-worker

# 搜索错误
sudo docker compose logs | grep -i error
```

### 资源监控

```bash
# 容器资源使用
sudo docker stats

# 服务器资源
free -h
df -h
```

### 健康检查

```bash
# 应用健康状态
curl http://localhost:8081/healthz

# 就绪状态
curl http://localhost:8081/readyz

# 容器状态
sudo docker compose ps
```

## 🎓 最佳实践

### 开发环境

1. **使用虚拟环境**
   ```bash
   uv sync
   source .venv/bin/activate
   ```

2. **本地运行服务**
   ```bash
   uv run uvicorn app.main:app --reload --port 5000
   ```

3. **本地运行 worker**
   ```bash
   uv run python -m app.worker
   ```

### 生产环境

1. **使用 Docker Compose**
   - 隔离环境
   - 易于扩展和管理

2. **配置反向代理**
   - 使用 Nginx
   - 启用 HTTPS
   - 配置负载均衡（如需要）

3. **设置监控**
   - 应用日志
   - 系统资源监控
   - 错误告警

4. **定期备份**
   - 数据库每日备份
   - 配置文件安全存储
   - 代码版本控制

## 🤝 团队协作

### Git 工作流

```bash
# 创建功能分支
git checkout -b feature/your-feature

# 提交更改
git add .
git commit -m "Add: your feature description"

# 推送到远程
git push origin feature/your-feature

# 创建 Pull Request
```

### 代码审查

- 遵循 Python PEP 8 风格指南
- 添加必要的注释和文档字符串
- 编写测试用例
- 更新相关文档

## 📞 支持和联系

### 获取帮助

- 📧 **Email**: team@teleport.computer
- 📚 **文档**: 查看项目中的所有 `.md` 文件
- 🐛 **Bug 报告**: 创建 GitHub Issue
- 💬 **讨论**: 使用团队沟通渠道

### 常用资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [SQLAlchemy 文档](https://docs.sqlalchemy.org/)
- [Docker 文档](https://docs.docker.com/)
- [AWS SES 文档](https://aws.amazon.com/ses/)
- [PostgreSQL 文档](https://www.postgresql.org/docs/)

## 🗺️ 路线图

### 短期目标

- [ ] 完成首次部署到测试环境
- [ ] 验证所有功能正常
- [ ] 根据需求调整业务逻辑
- [ ] 编写测试用例

### 中期目标

- [ ] 部署到生产环境
- [ ] 配置监控和告警
- [ ] 性能优化
- [ ] 添加新功能

### 长期目标

- [ ] 完善文档
- [ ] 自动化测试和部署（CI/CD）
- [ ] 扩展和优化架构
- [ ] 社区反馈和迭代

## 📝 更新日志

### v1.0.0 (2026-01-31)

**迁移完成**
- ✅ 从 TikTok Wrapped Backend 迁移配置
- ✅ 创建完整的文档体系
- ✅ 提供部署脚本和工具
- ✅ 配置多项目共存支持

**新增文档**
- QUICK_START.md - 快速开始指南
- DEPLOYMENT.md - 完整部署指南
- DEPLOYMENT_CHECKLIST.md - 部署检查清单
- PROJECT_COMPARISON.md - 项目对比
- MIGRATION_SUMMARY.md - 迁移总结
- PROJECT_OVERVIEW.md - 项目概览
- nginx.conf.example - Nginx 配置示例
- .env.template - 详细环境变量模板

**配置更新**
- docker-compose.yml - 更新项目名和容器名
- .env.example - 更新端口和数据库配置
- README.md - 完善文档和说明

**新增工具**
- deploy.sh - 交互式部署脚本

---

## 🎉 总结

Weekly Wrapped Newsletter Backend 现在已经准备好部署！所有必要的配置、文档和工具都已就绪。

**下一步**：
1. 选择合适的文档开始（建议从 [QUICK_START.md](./QUICK_START.md) 开始）
2. 配置环境变量
3. 运行部署脚本或按文档手动部署
4. 验证功能正常
5. 根据需求调整业务逻辑

**祝你部署顺利！** 🚀

如有任何问题，请查阅相关文档或联系技术支持团队。
