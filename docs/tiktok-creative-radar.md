# TikTok Creative Radar（周报趋势拉取）

周报的全球趋势（creator/sound/hashtag）来自 TikTok Creative Radar API。

## 现在的方式（不再使用 Playwright）

- Header（`Cookie` / `user-sign` / `web-id`）由管理员手动录入数据库。
- 提供管理页面手动更新 Header，并手动触发本周 trend 抓取。
- 定时周报批处理时，如果本周已存在 trend 数据，会直接复用，不再重复抓取。

## 管理页面

打开：

- `GET /admin/weekly-report/trends/page`
- 先输入 `.env` 里的 `TIKTOK_TRENDS_ADMIN_PASSWORD` 登录

页面支持：

- 加载当前 Header 配置
- 保存 Header 配置
- 手动抓取本周 Trends（仅抓趋势，不触发整条批处理流水线）
- 手动导入 Postman 的 creator/sound/hashtag 分页响应（自动解析并写入本周趋势）

## 相关接口

- `GET /admin/weekly-report/trends/headers`
- `PUT /admin/weekly-report/trends/headers`
- `POST /admin/weekly-report/trends/fetch`
- `POST /admin/weekly-report/trends/manual-import`
- `GET /admin/weekly-report/trends/stats/{global_report_id}`

以上接口都需要 `WEEKLY_TOKEN`（`Authorization: Bearer <token>`）。

## 定时批处理行为

- `POST /admin/cron/weekly-report-batch` 仍会入队 `weekly_report_fetch_trends -> weekly_report_batch_fetch -> ...`
- 但 `weekly_report_fetch_trends` 现在会先检查本周 `global_report_id` 是否已有完整 trends：
  - 已有：跳过抓取，直接进入下一步
  - 没有：才调用 TikTok API 抓取并落库

## 环境变量（可选兜底）

仍支持 env 作为兜底来源：

- `TIKTOK_ADS_COOKIE`
- `TIKTOK_ADS_USER_SIGN`
- `TIKTOK_ADS_WEB_ID`
- `TIKTOK_ADS_COUNTRY_CODE`（默认 `US`）

优先级：数据库配置 > 环境变量 > 内置匿名 Header。
