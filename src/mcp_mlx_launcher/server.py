import asyncio
from typing import Any

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

from mcp_mlx_launcher.process_manager import MlxProcessManager

# サーバーインスタンスとプロセス管理インスタンスの初期化
server = Server("mcp-mlx-launcher")
process_manager = MlxProcessManager()


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """AIエージェントに提供するツールの一覧とスキーマを定義します"""
    return [
        types.Tool(
            name="check_llm_status",
            description="指定されたポートでサーバーがリッスンしているか（稼働中か）を確認します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "port": {"type": "integer", "description": "確認するポート番号"}
                },
                "required": ["port"],
            },
        ),
        types.Tool(
            name="launch_llm_server",
            description="mlx_lm.server をサブプロセスとしてバックグラウンドで起動します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "起動するモデル名 (例: mlx-community/Llama-3-8B-Instruct-4bit)",
                    },
                    "port": {"type": "integer", "description": "サーバーを起動するポート番号"},
                },
                "required": ["model_name", "port"],
            },
        ),
        types.Tool(
            name="shutdown_llm_server",
            description="指定されたポートで稼働しているローカル LLM サーバープロセスを安全に終了させます。",
            inputSchema={
                "type": "object",
                "properties": {
                    "port": {"type": "integer", "description": "終了させるサーバーのポート番号"}
                },
                "required": ["port"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """AIエージェントから呼び出されたツールを実際に実行します"""
    if arguments is None:
        raise ValueError("Arguments are required")

    if name == "check_llm_status":
        port = arguments.get("port")
        if not isinstance(port, int):
            raise ValueError("Port must be an integer")
        
        is_running = process_manager.is_port_in_use(port)
        # MCPの仕様上、文字列で結果を返すのが一般的です ("true" or "false")
        return [types.TextContent(type="text", text=str(is_running).lower())]

    elif name == "launch_llm_server":
        model_name = arguments.get("model_name")
        port = arguments.get("port")
        if not isinstance(model_name, str) or not isinstance(port, int):
            raise ValueError("Invalid arguments for launch_llm_server")
        
        result_msg = process_manager.launch_server(model_name, port)
        return [types.TextContent(type="text", text=result_msg)]

    elif name == "shutdown_llm_server":
        port = arguments.get("port")
        if not isinstance(port, int):
            raise ValueError("Port must be an integer")
        
        result_msg = process_manager.shutdown_server(port)
        return [types.TextContent(type="text", text=result_msg)]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def run():
    """標準入出力(stdio)を利用してMCPサーバーを起動します"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-mlx-launcher",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    """パッケージのエントリーポイント"""
    asyncio.run(run())


if __name__ == "__main__":
    main()