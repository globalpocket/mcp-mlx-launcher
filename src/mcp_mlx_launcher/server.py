import asyncio
import json
from typing import Any

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from huggingface_hub import HfApi

from mcp_mlx_launcher.process_manager import MlxProcessManager

server = Server("mcp-mlx-launcher")
process_manager = MlxProcessManager()


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """AIエージェントに提供するツールの一覧とスキーマを定義します"""
    return [
        types.Tool(
            name="check_system_environment",
            description="現在のシステム環境（Apple Siliconか、空きメモリが何GBあるかなど）を診断します。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
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
            name="search_mlx_models",
            description="Hugging Faceからダウンロード可能なMLXフォーマットのLLMモデルを検索・リストアップします。",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "検索キーワード（例: 'llama', 'qwen'）。未指定の場合は人気のMLXモデルを返します。"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "取得する最大件数。デフォルトは10。"
                    }
                },
            },
        ),
        types.Tool(
            name="download_model",
            description="Hugging Faceから指定されたMLXモデルを事前にダウンロードし、ローカルにキャッシュします。大きなモデルの起動前の準備に利用します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "ダウンロードするモデル名 (例: mlx-community/Llama-3-8B-Instruct-4bit)"
                    }
                },
                "required": ["model_name"],
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
            name="restart_llm_server",
            description="指定されたポートで稼働しているサーバーを一度停止し、再起動します。モデルの切り替えなどにも使用できます。",
            inputSchema={
                "type": "object",
                "properties": {
                    "port": {"type": "integer", "description": "再起動するサーバーのポート番号"},
                    "model_name": {
                        "type": "string",
                        "description": "（オプション）新しく起動するモデル名。省略した場合は現在そのポートで稼働しているモデルをそのまま再起動します。"
                    },
                    "memory_requirement_gb": {
                        "type": "number",
                        "description": "（オプション）起動に必要な空きメモリの目安(GB)。未指定時はデフォルトで 4.0GB。"
                    }
                },
                "required": ["port"],
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
    if arguments is None and name not in ("list_running_servers", "search_mlx_models", "check_system_environment"):
        raise ValueError("Arguments are required")

    if name == "check_system_environment":
        info = process_manager.get_system_info()
        return [types.TextContent(type="text", text=json.dumps(info, indent=2))]

    elif name == "check_llm_status":
        port = arguments.get("port")
        if not isinstance(port, int):
            raise ValueError("Port must be an integer")
        
        is_running = process_manager.is_port_in_use(port)
        return [types.TextContent(type="text", text=str(is_running).lower())]

    elif name == "list_running_servers":
        servers = process_manager.get_running_servers()
        if not servers:
            return [types.TextContent(type="text", text="No running servers found.")]
        return [types.TextContent(type="text", text=json.dumps(servers, indent=2))]

    elif name == "search_mlx_models":
        query = arguments.get("search_query") if arguments else None
        limit = arguments.get("limit", 10) if arguments else 10

        if limit is not None and not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        if query is not None and not isinstance(query, str):
            raise ValueError("search_query must be a string")

        def _search():
            api = HfApi()
            models = api.list_models(
                search=query if query else None,
                tags="mlx",
                sort="downloads",
                direction=-1,
                limit=limit
            )
            results = []
            for m in models:
                results.append({
                    "modelId": m.id,
                    "downloads": getattr(m, 'downloads', 0),
                    "likes": getattr(m, 'likes', 0)
                })
            return results

        try:
            results = await asyncio.to_thread(_search)
            if not results:
                return [types.TextContent(type="text", text="No MLX models found.")]
            return [types.TextContent(type="text", text=json.dumps(results, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error searching models: {str(e)}")]

    elif name == "download_model":
        model_name = arguments.get("model_name")
        if not isinstance(model_name, str):
            raise ValueError("model_name must be a string")
            
        result_msg = await asyncio.to_thread(process_manager.download_model, model_name)
        return [types.TextContent(type="text", text=result_msg)]

    elif name == "launch_llm_server":
        model_name = arguments.get("model_name")
        port = arguments.get("port")
        memory_requirement_gb = arguments.get("memory_requirement_gb", 4.0)
        
        if not isinstance(model_name, str) or not isinstance(port, int):
            raise ValueError("Invalid arguments for launch_llm_server")
        if not isinstance(memory_requirement_gb, (int, float)):
            raise ValueError("memory_requirement_gb must be a number")
            
        result_msg = await asyncio.to_thread(
            process_manager.launch_server, 
            model_name, 
            port, 
            10, 
            float(memory_requirement_gb)
        )
        return [types.TextContent(type="text", text=result_msg)]

    elif name == "restart_llm_server":
        port = arguments.get("port")
        model_name = arguments.get("model_name")
        memory_requirement_gb = arguments.get("memory_requirement_gb", 4.0)
        
        if not isinstance(port, int):
            raise ValueError("Port must be an integer")
        if model_name is not None and not isinstance(model_name, str):
            raise ValueError("model_name must be a string")
        if not isinstance(memory_requirement_gb, (int, float)):
            raise ValueError("memory_requirement_gb must be a number")
            
        result_msg = await asyncio.to_thread(
            process_manager.restart_server, 
            port,
            model_name,
            10, 
            float(memory_requirement_gb)
        )
        return [types.TextContent(type="text", text=result_msg)]

    elif name == "shutdown_llm_server":
        port = arguments.get("port")
        if not isinstance(port, int):
            raise ValueError("Port must be an integer")
            
        result_msg = await asyncio.to_thread(process_manager.shutdown_server, port)
        return [types.TextContent(type="text", text=result_msg)]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def run():
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mcp-mlx-launcher",
                    server_version="0.2.1",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        active_servers = process_manager.get_running_servers()
        for port_str in active_servers.keys():
            try:
                process_manager.shutdown_server(int(port_str))
            except Exception:
                pass


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()