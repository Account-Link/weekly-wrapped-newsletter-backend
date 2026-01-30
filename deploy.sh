#!/bin/bash
# Weekly Wrapped Newsletter Backend - 部署脚本

set -e

echo "=================================="
echo "Weekly Wrapped Newsletter Backend"
echo "部署脚本 / Deployment Script"
echo "=================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在项目根目录
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    exit 1
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}警告: .env 文件不存在${NC}"
    echo "正在从 .env.example 复制..."
    cp .env.example .env
    echo -e "${GREEN}已创建 .env 文件，请编辑并填入必要的配置${NC}"
    echo ""
    echo "关键配置项："
    echo "  - PORT=8081 (避免与 TikTok Wrapped 的 8080 冲突)"
    echo "  - DATABASE_URL (使用独立的数据库名)"
    echo "  - SECRET_KEY (使用 'openssl rand -hex 32' 生成)"
    echo "  - ARCHIVE_BASE_URL 和 ARCHIVE_API_KEY"
    echo "  - AWS 相关配置"
    echo ""
    read -p "按回车继续编辑 .env 文件..." 
    ${EDITOR:-nano} .env
fi

# 检查端口配置
PORT=$(grep "^PORT=" .env | cut -d '=' -f2)
if [ -z "$PORT" ]; then
    PORT=8080
fi

if [ "$PORT" = "8080" ]; then
    echo -e "${YELLOW}警告: 端口设置为 8080，可能与 TikTok Wrapped 冲突${NC}"
    read -p "是否继续？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 询问操作类型
echo "请选择操作："
echo "1) 首次部署 (运行迁移 + 启动服务)"
echo "2) 更新部署 (重新构建 + 重启服务)"
echo "3) 仅运行迁移"
echo "4) 仅重启服务"
echo "5) 查看日志"
echo "6) 停止服务"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo -e "${GREEN}开始首次部署...${NC}"
        
        # 检查数据库连接
        echo "检查数据库配置..."
        DATABASE_URL=$(grep "^DATABASE_URL=" .env | cut -d '=' -f2)
        if [ -z "$DATABASE_URL" ]; then
            echo -e "${RED}错误: DATABASE_URL 未配置${NC}"
            exit 1
        fi
        
        # 运行迁移
        echo "运行数据库迁移..."
        sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"
        
        # 启动服务
        echo "启动服务..."
        sudo docker compose up -d --build \
            --scale cron-worker=2 \
            --scale cron-worker-watch=5 \
            --scale cron-worker-auth=2 \
            web cron-worker cron-worker-watch cron-worker-auth
        
        echo -e "${GREEN}部署完成！${NC}"
        echo "服务运行在端口: $PORT"
        echo "健康检查: curl http://localhost:$PORT/healthz"
        ;;
        
    2)
        echo -e "${GREEN}开始更新部署...${NC}"
        
        # 拉取最新代码（如果是 git 仓库）
        if [ -d ".git" ]; then
            echo "拉取最新代码..."
            git pull
        fi
        
        # 运行迁移
        echo "运行数据库迁移..."
        sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"
        
        # 重新构建并启动
        echo "重新构建并启动服务..."
        sudo docker compose down
        sudo docker compose up -d --build \
            --scale cron-worker=2 \
            --scale cron-worker-watch=5 \
            --scale cron-worker-auth=2 \
            web cron-worker cron-worker-watch cron-worker-auth
        
        echo -e "${GREEN}更新完成！${NC}"
        ;;
        
    3)
        echo -e "${GREEN}运行数据库迁移...${NC}"
        sudo docker compose run --rm web sh -lc "uv run alembic upgrade head"
        echo -e "${GREEN}迁移完成！${NC}"
        ;;
        
    4)
        echo -e "${GREEN}重启服务...${NC}"
        sudo docker compose restart
        echo -e "${GREEN}重启完成！${NC}"
        ;;
        
    5)
        echo "查看日志 (按 Ctrl+C 退出)..."
        echo ""
        read -p "选择日志类型 (1=全部, 2=web, 3=worker): " log_choice
        case $log_choice in
            1) sudo docker compose logs -f ;;
            2) sudo docker compose logs -f web ;;
            3) sudo docker compose logs -f cron-worker ;;
            *) echo -e "${RED}无效选项${NC}" ;;
        esac
        ;;
        
    6)
        echo -e "${YELLOW}停止服务...${NC}"
        sudo docker compose down
        echo -e "${GREEN}服务已停止${NC}"
        ;;
        
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "常用命令："
echo "  查看状态: sudo docker compose ps"
echo "  查看日志: sudo docker compose logs -f"
echo "  进入容器: sudo docker compose exec web sh"
echo "  停止服务: sudo docker compose down"
echo "=================================="
