# 将 UniKB 接入 MCP 客户端

UniKB 的 MCP server 使用 stdio 传输。先在 `backend` 目录安装依赖并配置至少一个 LLM API Key，然后验证启动：

```bash
cd backend
python -m app.mcp.server default
```

进程会等待 MCP 客户端的 stdio 请求；这是正常行为。客户端配置里的 `cwd` 必须改成你本机 `backend` 的绝对路径。

## Claude Desktop / Cursor / Trae

将下列对象合并到对应客户端的 MCP 配置中：

```json
{
  "mcpServers": {
    "unikb": {
      "command": "python",
      "args": ["-m", "app.mcp.server", "default"],
      "cwd": "/absolute/path/to/UniKB/backend",
      "env": {
        "PYTHONPATH": "/absolute/path/to/UniKB/backend"
      }
    }
  }
}
```

Windows 路径示例：`C:\\work\\UniKB\\backend`。保存配置后重启客户端，工具列表中应出现 `hybrid_search`、`calculator` 和 `current_date`。

> 请勿将 `.env` 或 API Key 提交到 Git。仓库中的 `.env.example` 仅用于生成本地配置。
