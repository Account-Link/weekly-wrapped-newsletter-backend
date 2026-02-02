# TikTok Creative Radar（周报趋势拉取）

周报的「全球趋势」（热门创作者、音乐、话题）来自 **TikTok Ads Creative Center** 的未公开接口（Creative Radar API）。

**默认无需登录**：未配置 `TIKTOK_ADS_COOKIE` / `TIKTOK_ADS_USER_SIGN` / `TIKTOK_ADS_WEB_ID` 时，会使用内置的匿名请求头（anonymous-user-id、user-sign、web-id、Cookie），可直接拉取趋势数据。若接口返回 **"no permission"**，再按下文配置登录态或刷新 Cookie。

## 定时任务发周报（与趋势的关系）

**周报定时任务可以照常跑，不依赖趋势接口。**

- 定时任务：调用 `POST /admin/cron/weekly-report-batch`（需 `WEEKLY_TOKEN`），会入队 `weekly_report_fetch_trends` → `weekly_report_batch_fetch` → 分析 → 发送。
- 若 **未配置** 或 **Creative Radar 返回 no permission**：worker 会打警告日志，但**不会中断流程**，仍会拉用户数据、分析、发邮件，只是本周「全球趋势」为空，报告里不展示趋势排行。
- 若希望**每周都带趋势**：可设置 `TIKTOK_ADS_SESSION_DIR` 让 worker **每次 fetch trend 时自动抓取 header**（见下文），或用手动/脚本方式刷新 Cookie 等。

## 现象

三个趋势接口（creator、sound、hashtag）都报 **no permission**：

- 接口：`https://ads.tiktok.com/creative_radar_api/v1/popular_trend/creator/list`、`.../sound/rank_list`、`.../hashtag/list`
- 鉴权方式：请求头里的 `Cookie`、`user-sign`、`web-id`（均为会话相关，会过期）

## 解决方法

### 1. 刷新会话凭证（最常见）

Cookie、user-sign、web-id 会过期，需要从**已登录的浏览器**里重新抓取并更新到环境变量。

**步骤：**

1. 用有 **TikTok for Business / 广告后台** 权限的账号登录：<https://ads.tiktok.com>  
2. 打开 **Creative Center → Inspiration → Popular**（任选 Creators / Music / Hashtags 任一页）：  
   - 例如：<https://ads.tiktok.com/business/creativecenter/inspiration/popular/creator/pc/en>
3. 打开浏览器 **开发者工具（F12）→ Network**，刷新页面或切换 Tab（如切到 Music、Hashtags）。
4. 在 Network 里找到请求：  
   - URL 包含 `creative_radar_api/v1/popular_trend/`
5. 点开该请求，在 **Request Headers** 里复制：
   - **Cookie**：整段 Cookie 字符串
   - **user-sign**：请求头 `user-sign` 的值
   - **web-id**：请求头 `web-id` 的值
6. 更新 `.env`（或部署环境变量）：
   ```bash
   TIKTOK_ADS_COOKIE="<粘贴 Cookie>"
   TIKTOK_ADS_USER_SIGN="<粘贴 user-sign>"
   TIKTOK_ADS_WEB_ID="<粘贴 web-id>"
   ```
7. 重启 worker（或重新部署），再跑一次周报拉趋势。

**注意：**

- Cookie 里可能包含敏感信息，不要提交到代码库，只放在 `.env` 或 CI/部署环境变量中。
- 会话通常几小时到几天会失效，若再次出现 "no permission"，按上述步骤重新抓取并更新。

### 半自动抓取（脚本 + 定时前刷新）

项目里提供了用 **Playwright** 的半自动抓取脚本，可定期在发周报前跑一次，把抓到的三个 header 写入文件或打印成 env，再更新到运行 worker 的环境。

**安装与运行：**

```bash
# 安装可选依赖 Playwright 与 Chromium（一次性）
uv sync --extra tiktok-radar-refresh
uv run playwright install chromium

# 有界面：打开浏览器，若未登录会跳转登录，登录后脚本会抓到 creative_radar 请求的 header 并输出
uv run python scripts/refresh_tiktok_creative_radar_headers.py

# 输出到文件（再合并进 .env 或由部署环境加载）
uv run python scripts/refresh_tiktok_creative_radar_headers.py -o .env.radar
```

**与定时任务配合：**

1. **方式 A（推荐）**：每周在发周报前（例如周一早上发周报，则周日晚上或周一早上）在本机或一台有浏览器的机器上跑一次脚本，把输出的三行 env 更新到**运行 worker 的环境**（.env 或 CI/部署环境变量），然后重启 worker（或等下次发周报时已生效）。
2. **方式 B**：若部署环境能跑浏览器（例如一台有 Chromium 的 Linux 机），可定时跑脚本并把输出写入 `.env.radar`，再在启动 worker 前 `source .env.radar` 或合并进主 `.env`。注意 TikTok 可能要求登录/验证码，**完全无人值守不一定可行**，多数情况下是「定时跑脚本 + 偶尔需要人工登录」。
3. **可选：保存会话**：第一次有界面运行时可加 `--persist ./radar_session`，脚本会把浏览器 session 存到该目录；下次可尝试 `--headless --persist ./radar_session` 无头刷新。若 TikTok 未要求重新登录，则可实现一定程度的自动化；若会话失效，仍需有界面登录一次。

**总结：** 定时任务发周报本身不依赖趋势；若希望带趋势，可用「每次 fetch 时抓取」或脚本定期刷新。

### 每次 fetch trend 时自动抓取 header（推荐）

设置 **`TIKTOK_ADS_SESSION_DIR`** 后，worker 在**每次执行 fetch_trends** 时会在**调用 fetch 接口之前**、在进程内用 Playwright 无头浏览器抓取 Cookie / user-sign / web-id，再调 Creative Radar API。**不需要单独跑脚本**，header 是在 worker 里、发请求前拿到的。脚本只用于一次性生成 `state.json`（见下）。

**步骤：**

1. **一次性：生成会话并保存**
   - 在本机（有图形界面）安装 Playwright 与 Chromium：`uv sync --extra tiktok-radar-refresh`，`uv run playwright install chromium`
   - 运行脚本并登录 TikTok Ads：`uv run python scripts/refresh_tiktok_creative_radar_headers.py --persist ./radar_session`
   - 浏览器打开后登录 ads.tiktok.com，脚本会抓到 header 并把 session 存到 `./radar_session/state.json`
2. **部署 worker 的环境**
   - 把 `radar_session` 目录放到 worker 能读到的路径（例如挂载到容器内 `/app/radar_session`）
   - 在 worker 环境安装 Playwright + Chromium（Docker 需在镜像里加 `playwright install chromium` 及依赖）
   - 设置环境变量：`TIKTOK_ADS_SESSION_DIR=/app/radar_session`（或你挂载的路径）
3. **之后**
   - 每次定时任务触发 `weekly_report_fetch_trends` 时，worker 会先加载 `state.json`、打开 Popular 页、抓取本次请求的 header 再调 API。抓取失败则回退到 `TIKTOK_ADS_COOKIE` 等 env。

**注意：**

- Worker 所在环境必须能跑 Chromium（内存与依赖足够）。Docker 需在 Dockerfile 中安装 Playwright 与 Chromium（见 [Playwright Docker](https://playwright.dev/python/docs/docker)）。
- 若未安装 Playwright 或 `TIKTOK_ADS_SESSION_DIR` 下没有有效的 `state.json`，会静默回退到 env 中的 Cookie/user-sign/web-id。
- 会话过期后（TikTok 要求重新登录），需再在本机有界面跑一次脚本 `--persist` 更新 `radar_session`，并重新挂载或同步到 worker 环境。

### 2. 确认账号与地区

- 使用的账号必须能正常打开 **TikTok Ads Manager** 和 **Creative Center**。
- 部分地区或账号类型可能无法使用 Creative Center，或接口返回 no permission；可换账号或地区（如 `TIKTOK_ADS_COUNTRY_CODE=US`）再试。

### 3. 不配置时的行为

若未配置 `TIKTOK_ADS_COOKIE` / `TIKTOK_ADS_USER_SIGN` / `TIKTOK_ADS_WEB_ID`，或接口持续返回错误（含 no permission），worker 会：

- 记录警告日志（如 `weekly_report_fetch_trends.api_error`）
- **不中断**周报流程：仍会执行 `weekly_report_batch_fetch` 及后续分析、发送，只是本周的「全球趋势」数据为空，用户报告里不会带趋势排行

因此不配置或暂时拉不到趋势，只会影响「是否展示趋势」，不会影响周报整体生成与发送。

## 环境变量

| 变量 | 说明 |
|------|------|
| `TIKTOK_ADS_SESSION_DIR` | 可选。目录路径，内含 `state.json`（由脚本 `--persist` 生成）。若设置，worker 每次 fetch_trends 会先尝试用 Playwright 抓 header，再调 API。 |
| `TIKTOK_ADS_COOKIE` | 可选。不设置时使用内置匿名 Cookie；若接口返回 no permission，可从浏览器复制 Cookie 覆盖。 |
| `TIKTOK_ADS_USER_SIGN` | 可选。不设置时使用内置匿名 user-sign。 |
| `TIKTOK_ADS_WEB_ID` | 可选。不设置时使用内置匿名 web-id。 |
| `TIKTOK_ADS_COUNTRY_CODE` | 可选，趋势地区，默认 `US` |

详见 `app/settings.py`、`app/services/tiktok_creative_radar_client.py`、`app/services/tiktok_creative_radar_capture.py`。
