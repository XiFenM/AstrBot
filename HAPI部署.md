# HAPI 部署全流程指南

> 适用环境：Linux 宿主机 / 多个 Docker 容器 / 手机远程控制 Claude Code

---

## 架构概览

HAPI 由三个独立进程组成，需要分别启动：

| 组件 | 运行位置 | 作用 |
|------|----------|------|
| Hub | 宿主机 | 监听 3006 端口，管理所有 session，是整个系统的中心 |
| CLI (hapi) | 每个 Docker 容器内 | 启动 Claude Code 并将 session 注册到 Hub |
| Runner | 容器内（可选） | 负责远程创建新 session，不启动不影响已有 session 使用 |
| cloudflared | 宿主机 | 将公网流量通过隧道转发到本地 Hub，无需开放防火墙端口 |

整体链路：

```
手机 (PWA) → hapi.yourdomain.com → Cloudflare → cloudflared 隧道 → 宿主机:3006 (Hub) → 各容器 (hapi CLI)
```

---

## 第一步：在宿主机启动 Hub

创建 systemd 服务文件，使 Hub 开机自启：

```bash
sudo nano /etc/systemd/system/hapi-hub.service
```

写入以下内容（将 `/usr/bin/hapi` 替换为实际路径，用 `which hapi` 查询）：

```ini
[Unit]
Description=HAPI Hub
After=network.target

[Service]
ExecStart=/usr/bin/hapi hub --no-relay
Restart=always
User=root
WorkingDirectory=/root

[Install]
WantedBy=multi-user.target
```

启动并设置开机自启：

```bash
sudo systemctl daemon-reload
sudo systemctl start hapi-hub
sudo systemctl enable hapi-hub
```

验证 Hub 正在监听：

```bash
ss -tlnp | grep 3006
```

> 看到 3006 端口有监听输出，说明 Hub 已正常运行。

---

## 第二步：配置 Cloudflare Tunnel

> 前置条件：需要一个 Cloudflare 账号，以及托管在 Cloudflare 上的域名。

### 安装 cloudflared

Debian / Ubuntu：

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
cloudflared -v   # 验证安装
```

### 登录并创建隧道

```bash
cloudflared tunnel login
```

浏览器打开终端输出的 URL，授权域名后，`~/.cloudflared/` 目录下会生成 `cert.pem`。

```bash
cloudflared tunnel create hapi
```

记下输出的 Tunnel UUID，格式为 `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`。

### 创建配置文件

```bash
nano ~/.cloudflared/config.yml
```

填入以下内容（替换 UUID 和域名）：

```yaml
tunnel: <你的tunnel-UUID>
credentials-file: /root/.cloudflared/<你的tunnel-UUID>.json
protocol: http2          # 必须，HAPI 用 SSE 需要 http2

ingress:
  - hostname: hapi.yourdomain.com
    service: http://localhost:3006
  - service: http_status:404
```

> `protocol: http2` 这行必须保留，否则 SSE 长连接会超时断开。

### 绑定 DNS 并设置为系统服务

```bash
cloudflared tunnel route dns hapi hapi.yourdomain.com

sudo mkdir -p /etc/cloudflared/
sudo mv ~/.cloudflared/config.yml /etc/cloudflared/
sudo cp ~/.cloudflared/*.json /etc/cloudflared/
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

---

## 第三步：在 Docker 容器内配置 hapi CLI

### 推荐的容器启动命令

```bash
docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --name aurimo-dev \
    -v ~/.claude:/root/.claude \
    -v ~/workspace/aurimo-dev:/root/workspace/aurimo-dev \
    sile-dev:v0 \
    bash

docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --name astrbot-dev \
    -v ~/.claude:/root/.claude \
    -v ~/workspace/astrbot-dev:/root/workspace/astrbot-dev \
    sile-dev:v0 \
    bash
```

> `--net host` 使容器与宿主机共享网络，容器内 `localhost` 就是宿主机，无需额外配置连接地址。

### 在容器内启动 hapi

设置容器名称（用于在手机 HAPI 界面中区分各容器）：

```bash
# 临时设置
HAPI_HOSTNAME="容器1-项目名" hapi

# 永久写入
echo 'export HAPI_HOSTNAME="容器1-项目名"' >> ~/.bashrc
source ~/.bashrc
HAPI_HOSTNAME=SERVER_HOST hapi runner start
```

---

## 第四步：验证与使用

1. 用手机浏览器访问 `https://hapi.yourdomain.com`
2. 看到 HAPI Web 界面，列出所有已连接的容器 session
3. 点进任意 session 即可远程控制该容器内的 Claude Code
4. 在手机浏览器中选择「添加到主屏幕」，作为 PWA 使用

---

## 重要注意事项

### 安全

- HAPI Web 界面默认无密码，任何人知道域名就能访问
- 建议在 Cloudflare Zero Trust 中开启 Access 做身份验证
- 或在 HAPI `settings.json` 中配置访问密码

### 对话记录管理

Claude Code 的完整对话记录存储在：

```
~/.claude/projects/<项目路径>/
```

- HAPI 界面只显示最近 500 条消息（MessageBuffer 限制），但不影响 Claude Code 实际使用的上下文
- 迁移对话记录：直接复制 `.jsonl` 文件到目标机器对应路径即可，无需特殊处理
- 随着对话变长，建议定期使用 `/compact` 压缩，并将重要信息维护到 `CLAUDE.md`

### 容器持久化

- 务必挂载 `-v ~/.claude:/root/.claude`，否则容器删除后对话记录丢失
- 项目代码也建议挂载到宿主机目录
- 容器内 hapi 进程退出后 session 消失，建议用 `supervisor` 或 `pm2` 保持常驻

### Runner 说明

Runner 启动时报 `ECONNREFUSED` 是因为找不到 Hub，只要 Hub 正常运行后会自动重试恢复。Runner 的作用是从手机端远程创建新 session，不启动 Runner 不影响已有 session 的正常使用。