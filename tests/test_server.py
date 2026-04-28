import pytest
import json
from unittest.mock import patch
from mcp_mlx_launcher.server import handle_list_tools, handle_call_tool


@pytest.mark.asyncio
async def test_handle_list_tools():
    """定義されているツールの一覧とスキーマが正しく返却されるか"""
    tools = await handle_list_tools()
    assert len(tools) == 4
    names = [tool.name for tool in tools]
    assert "check_llm_status" in names
    assert "list_running_servers" in names
    assert "launch_llm_server" in names
    assert "shutdown_llm_server" in names


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.process_manager.get_running_servers")
async def test_list_running_servers(mock_get_servers):
    """稼働中サーバー一覧取得ツールのテスト"""
    mock_get_servers.return_value = {"8080": {"pid": 123, "model": "test"}}
    result = await handle_call_tool("list_running_servers", {})
    assert "123" in result[0].text
    assert "test" in result[0].text

    mock_get_servers.return_value = {}
    result = await handle_call_tool("list_running_servers", {})
    assert result[0].text == "No running servers found."


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.process_manager.is_port_in_use")
async def test_check_llm_status_success(mock_is_port_in_use):
    """ポートステータス確認ツールのテスト"""
    mock_is_port_in_use.return_value = True
    result = await handle_call_tool("check_llm_status", {"port": 8080})
    assert result[0].text == "true"

    mock_is_port_in_use.return_value = False
    result = await handle_call_tool("check_llm_status", {"port": 8080})
    assert result[0].text == "false"


@pytest.mark.asyncio
async def test_check_llm_status_invalid_args():
    """引数バリデーションテスト (check_llm_status)"""
    # 引数なし
    with pytest.raises(ValueError, match="Arguments are required"):
        await handle_call_tool("check_llm_status", None)
        
    # 型が不正
    with pytest.raises(ValueError, match="Port must be an integer"):
        await handle_call_tool("check_llm_status", {"port": "8080"})


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.process_manager.launch_server")
async def test_launch_llm_server_success(mock_launch_server):
    """起動ツールのテスト (スレッド逃がしと追加引数が正しく渡るか)"""
    mock_launch_server.return_value = "Launched successfully"
    result = await handle_call_tool(
        "launch_llm_server", {"model_name": "test-model", "port": 8080, "memory_requirement_gb": 8.5}
    )
    assert result[0].text == "Launched successfully"
    # デフォルトのタイムアウト10秒と、指定した8.5GBが正しく渡されているか
    mock_launch_server.assert_called_once_with("test-model", 8080, 10, 8.5)


@pytest.mark.asyncio
async def test_launch_llm_server_invalid_args():
    """引数バリデーションテスト (launch_llm_server)"""
    # 必須パラメータ不足
    with pytest.raises(ValueError, match="Invalid arguments for launch_llm_server"):
        await handle_call_tool("launch_llm_server", {"model_name": "test-model"})
        
    # 型が不正
    with pytest.raises(ValueError, match="Invalid arguments for launch_llm_server"):
        await handle_call_tool("launch_llm_server", {"model_name": 123, "port": "8080"})

    # メモリ要求量の型が不正
    with pytest.raises(ValueError, match="memory_requirement_gb must be a number"):
        await handle_call_tool("launch_llm_server", {"model_name": "test", "port": 8080, "memory_requirement_gb": "8"})


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.process_manager.shutdown_server")
async def test_shutdown_llm_server_success(mock_shutdown_server):
    """停止ツールのテスト"""
    mock_shutdown_server.return_value = "Shut down successfully"
    result = await handle_call_tool("shutdown_llm_server", {"port": 8080})
    assert result[0].text == "Shut down successfully"
    mock_shutdown_server.assert_called_once_with(8080)


@pytest.mark.asyncio
async def test_shutdown_llm_server_invalid_args():
    """引数バリデーションテスト (shutdown_llm_server)"""
    with pytest.raises(ValueError, match="Port must be an integer"):
        await handle_call_tool("shutdown_llm_server", {"port": "8080"})


@pytest.mark.asyncio
async def test_unknown_tool():
    """未知のツール呼び出しテスト"""
    with pytest.raises(ValueError, match="Unknown tool: invalid_tool"):
        await handle_call_tool("invalid_tool", {})