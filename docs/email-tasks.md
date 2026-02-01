# 邮件发送任务开关说明

本项目的邮件发送由后台 worker 执行任务完成。为了防止部署后自动发邮件，
我们在 `docker-compose.yml` 中禁用了 `email_send` 与 `weekly_report_send` 两个任务。

## 当前状态（默认禁用）

在 `docker-compose.yml` 的 `cron-worker` 环境变量中：

- `WORKER_TASK_ALLOW` 不包含：
  - `email_send`（wrapped 邮件）
  - `weekly_report_send`（周报邮件）

因此即便有任务入队，worker 也不会执行发送。

## 如何恢复邮件发送

1. 编辑 `docker-compose.yml`
2. 在 `cron-worker.environment` 的 `WORKER_TASK_ALLOW` 中加入：
   - `email_send`
   - `weekly_report_send`
3. 重新部署

示例（只展示一行，按现有顺序插入即可）：

```
WORKER_TASK_ALLOW=wrapped_analysis,email_send,watch_history_verify,reauth_notify,...,weekly_report_analyze,weekly_report_send
```

## 额外说明

- 如果你不希望自动触发 wrapped 生成，可在 `.env` 中设置：
  ```
  DISABLE_AUTO_WRAPPED=true
  ```
  这会阻止自动创建 wrapped 任务（不影响手动触发）。

