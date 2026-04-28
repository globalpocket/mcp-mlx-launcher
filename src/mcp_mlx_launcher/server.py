import asyncio
import json
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
            name="list_running_servers",
            description="現在バックグラウンドで稼働しているすべてのローカルLLMサーバー（ポート番号とモデル名）の一覧を取得します。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="launch_llm_server",
            description="mlx_lm.server をサブプロセスとしてバックグラウンドで起動します。空きメモリが少ない場合は起動が拒否されます。",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "起動するモデル名 (例: mlx-community/Llama-3-8B-Instruct-4bit)",
                    },
                    "port": {"type": "integer", "description": "サーバーを起動するポート番号"},
                    "memory_requirement_gb": {
                        "type": "number",
                        "description": "起動に必要な空きメモリの目安(GB)。未指定時はデフォルトで 4.0GB。"
                    }
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
    if arguments is None and name != "list_running_servers":
        raise ValueError("Arguments are required")

    if name == "check_llm_status":
        port = arguments.get("port")
        if not isinstance(port, int):
            raise ValueError("Port must be an integer")
        
        is_running = process_manager.is_port_in_use(port)
        return [types.TextContent(type="text", text=str(is_running).lower())]

    elif name == "list_running_servers":
        servers = process_manager.get_running_servers()
        if not servers:
            return [types.TextContent(type="text", text="No running servers found.")]
        # JSON文字列としてフォーマットして返す
        return [types.TextContent(type="text", text=json.dumps(servers, indent=2))]

    elif name == "launch_llm_server":
        model_name = arguments.get("model_name")
        port = arguments.get("port")
        memory_requirement_gb = arguments.get("memory_requirement_gb", 4.0)
        
        if not isinstance(model_name, str) or not isinstance(port, int):
            raise ValueError("Invalid arguments for launch_llm_server")
        if not isinstance(memory_requirement_gb, (int, float)):
            raise ValueError("memory_requirement_gb must be a number")
            
        # イベントループのブロッキングを防ぐため別スレッドで実行
        result_msg = await asyncio.to_thread(
            process_manager.launch_server, 
            model_name, 
            port, 
            10, 
            float(memory_requirement_gb)
        )
        return [types.TextContent(type="text", text=result_msg)]

    elif name == "shutdown_llm_server":
        port = arguments.get("port")
        if not isinstance(port, int):
            raise ValueError("Port must be an integer")
            
        # イベントループのブロッキングを防ぐため別スレッドで実行
        result_msg = await asyncio.to_thread(process_manager.shutdown_server, port)
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
                server_version="0.1.1",
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