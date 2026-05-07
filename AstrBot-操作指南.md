# AstrBot 操作指南

## 服务管理

```bash
# 查看运行状态
systemctl status astrbot

# 启动
systemctl start astrbot

# 停止
systemctl stop astrbot

# 重启
systemctl restart astrbot

# 开机自启（已启用）
systemctl enable astrbot

# 取消开机自启
systemctl disable astrbot
```

## 日志查看

```bash
# 实时跟踪日志
tail -f /root/workspace/host/astrbot.log

# 查看最近 100 行
tail -100 /root/workspace/host/astrbot.log

# 搜索关键字（如错误信息）
grep -i "error" /root/workspace/host/astrbot.log
```

## 数据目录

| 路径 | 说明 |
|------|------|
| `/root/workspace/host/AstrBot/data/data_v4.db` | 会话与对话历史（SQLite） |
| `/root/workspace/host/AstrBot/data/config/` | 配置文件 |
| `/root/workspace/host/AstrBot/data/plugins/` | 插件数据 |
| `/root/workspace/host/astrbot.log` | 运行日志 |

## Dashboard

浏览器访问：http://localhost:6185

## 手动启动（不使用 systemd）

```bash
cd /root/workspace/host/AstrBot
/root/.local/bin/uv run python main.py
```

---

## HAPI Hub（Claude Code 会话管理）

### 服务管理

```bash
# 查看运行状态
systemctl status hapi

# 启动
systemctl start hapi

# 停止
systemctl stop hapi

# 重启
systemctl restart hapi
```

### 日志查看

```bash
# 实时跟踪日志
tail -f /root/workspace/host/hapi.log

# 查看最近 100 行
tail -100 /root/workspace/host/hapi.log
```

### 手动启动（不使用 systemd）

```bash
hapi hub --no-relay
```

### 源码开发与更新

HAPI 源码仓库位于 `/root/workspace/host/hapi`（从 fork 仓库 clone）。

```bash
# 修改代码后，重新构建并安装
cd /root/workspace/host/hapi
~/.bun/bin/bun run build:single-exe && cp cli/dist-exe/bun-linux-x64-baseline/hapi /usr/local/bin/hapi

# 重启 Hub 使更新生效
systemctl restart hapi

# 更新容器内的 Runner（也需要用新二进制）
docker cp /usr/local/bin/hapi aurimo-dev:/usr/local/bin/hapi
docker exec aurimo-dev bash -lc "/usr/local/bin/hapi runner start-sync" &
```

**回退到 npm 版本：**
```bash
# 宿主机
rm /usr/local/bin/hapi
# 容器内
docker exec aurimo-dev rm /usr/local/bin/hapi
# 两边都会自动 fallback 到 npm 安装的版本
```

**依赖安装（首次或依赖变更后）：**
```bash
cd /root/workspace/host/hapi
~/.bun/bin/bun install
```

### HAPI Runner（容器内）

Runner 运行在 Docker 容器 `aurimo-dev` 中，通过 WebSocket 连接 Hub。

```bash
# 查看 Runner 进程
docker exec aurimo-dev bash -c "ps aux | grep hapi | grep -v grep"

# 启动 Runner（必须用 bash -lc 以加载环境变量）
docker exec -d aurimo-dev bash -lc "/usr/local/bin/hapi runner start-sync"

# 停止 Runner
docker exec aurimo-dev bash -lc "/usr/local/bin/hapi runner stop"

# 查看 Runner 日志
docker exec aurimo-dev cat /root/.hapi/logs/runner.log
```

**容器内环境变量**（配置在 `/root/.bashrc` 中）：
| 变量 | 值 | 说明 |
|------|-----|------|
| `CLI_API_TOKEN` | `SXplljDYt5BKwokp45IDYymUHkr1HqMR17drdWLwLg8` | Hub 认证 token |
| `HAPI_API_URL` | `http://172.17.0.1:3006` | Hub 地址（docker 宿主机） |
| `HAPI_HOSTNAME` | `aurimo-dev-us` | 机器显示名 |

**容器内机器 ID：** `e41f502b-41cf-4260-ab8e-7fd3c2d14b61`（存储在 `/root/.hapi/settings.json`）

### Web API 认证

HAPI Web API 使用 JWT 认证，不能直接用 cliApiToken。需要两步：

```bash
# 1. 用 cliApiToken 换取 JWT
JWT=$(curl -s -X POST http://localhost:3006/api/auth \
  -H "Content-Type: application/json" \
  -d '{"accessToken": "SXplljDYt5BKwokp45IDYymUHkr1HqMR17drdWLwLg8"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# 2. 用 JWT 调用 API
curl -s -H "Authorization: Bearer $JWT" http://localhost:3006/api/machines
curl -s -H "Authorization: Bearer $JWT" "http://localhost:3006/api/machines/<machine_id>/directory?path=/root"
```

### 注意事项

- **Hub 重启后 Runner 会断连退出**，需要手动重启容器内的 Runner
- **docker exec 不加载 .bashrc**，必须用 `bash -lc "..."` 包裹命令，否则环境变量缺失会导致 `CLI_API_TOKEN is required` 报错
- **Hub 端口为 3006**（Hono 框架），容器内 CLI 的 fastify 端口为 36033，注意区分
- **多台机器在线时注意区分 Machine ID**，可通过 `GET /api/machines` 查看所有机器及其 hostname

### 数据目录

| 路径 | 说明 |
|------|------|
| `/root/workspace/host/hapi.log` | Hub 运行日志 |
| `/root/workspace/host/hapi/` | 源码仓库 |
| `/usr/local/bin/hapi` | 源码构建的可执行文件 |
| `/usr/bin/hapi` | npm 安装的原始版本 |
| `/etc/systemd/system/hapi.service` | Hub 服务配置文件 |
| `/root/.hapi/settings.json` | Hub 配置（含 cliApiToken） |

---

## 服务配置文件

```
/etc/systemd/system/astrbot.service
/etc/systemd/system/hapi.service
```
