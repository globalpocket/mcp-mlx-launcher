import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from mcp_mlx_launcher.server import handle_list_tools, handle_call_tool, run


@pytest.mark.asyncio
async def test_handle_list_tools():
    """定義されているツールの一覧とスキーマが正しく返却されるか"""
    tools = await handle_list_tools()
    assert len(tools) == 8 # 4から8に変更
    names = [tool.name for tool in tools]
    assert "check_system_environment" in names
    assert "check_llm_status" in names
    assert "list_running_servers" in names
    assert "search_mlx_models" in names
    assert "download_model" in names
    assert "launch_llm_server" in names
    assert "restart_llm_server" in names
    assert "shutdown_llm_server" in names


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.process_manager.get_system_info")
async def test_check_system_environment(mock_get_sys_info):
    """システム環境診断ツールのテスト"""
    mock_get_sys_info.return_value = {"total_memory_gb": 16.0}
    result = await handle_call_tool("check_system_environment", {})
    assert "16.0" in result[0].text


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
    with pytest.raises(ValueError, match="Arguments are required"):
        await handle_call_tool("check_llm_status", None)
        
    with pytest.raises(ValueError, match="Port must be an integer"):
        await handle_call_tool("check_llm_status", {"port": "8080"})


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.HfApi")
async def test_search_mlx_models_success(mock_hf_api):
    """モデル検索ツールの成功テスト"""
    mock_api_instance = MagicMock()
    mock_model = MagicMock()
    mock_model.id = "mlx-community/test-model"
    mock_model.downloads = 1000
    mock_model.likes = 50
    mock_api_instance.list_models.return_value = [mock_model]
    mock_hf_api.return_value = mock_api_instance
    
    result = await handle_call_tool("search_mlx_models", {"search_query": "test", "limit": 5})
    assert "mlx-community/test-model" in result[0].text
    mock_api_instance.list_models.assert_called_once_with(
        search="test", tags="mlx", sort="downloads", direction=-1, limit=5
    )


@pytest.mark.asyncio
async def test_search_mlx_models_invalid_args():
    """引数バリデーションテスト (search_mlx_models)"""
    with pytest.raises(ValueError, match="limit must be an integer"):
        await handle_call_tool("search_mlx_models", {"limit": "5"})
    with pytest.raises(ValueError, match="search_query must be a string"):
        await handle_call_tool("search_mlx_models", {"search_query": 123})


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.process_manager.download_model")
async def test_download_model_success(mock_download_model):
    """ダウンロードツールの成功テスト"""
    mock_download_model.return_value = "Successfully downloaded and cached model: test-model"
    result = await handle_call_tool("download_model", {"model_name": "test-model"})
    assert result[0].text == "Successfully downloaded and cached model: test-model"
    mock_download_model.assert_called_once_with("test-model")


@pytest.mark.asyncio
async def test_download_model_invalid_args():
    """引数バリデーションテスト (download_model)"""
    with pytest.raises(ValueError, match="model_name must be a string"):
        await handle_call_tool("download_model", {"model_name": 123})
    with pytest.raises(ValueError, match="Arguments are required"):
        await handle_call_tool("download_model", None)


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.process_manager.launch_server")
async def test_launch_llm_server_success(mock_launch_server):
    """起動ツールのテスト"""
    mock_launch_server.return_value = "Launched successfully"
    result = await handle_call_tool(
        "launch_llm_server", {"model_name": "test-model", "port": 8080, "memory_requirement_gb": 8.5}
    )
    assert result[0].text == "Launched successfully"
    mock_launch_server.assert_called_once_with("test-model", 8080, 10, 8.5)


@pytest.mark.asyncio
async def test_launch_llm_server_invalid_args():
    """引数バリデーションテスト (launch_llm_server)"""
    with pytest.raises(ValueError, match="Invalid arguments for launch_llm_server"):
        await handle_call_tool("launch_llm_server", {"model_name": "test-model"})
        
    with pytest.raises(ValueError, match="Invalid arguments for launch_llm_server"):
        await handle_call_tool("launch_llm_server", {"model_name": 123, "port": "8080"})

    with pytest.raises(ValueError, match="memory_requirement_gb must be a number"):
        await handle_call_tool("launch_llm_server", {"model_name": "test", "port": 8080, "memory_requirement_gb": "8"})


@pytest.mark.asyncio
@patch("mcp_mlx_launcher.server.process_manager.restart_server")
async def test_restart_llm_server_success(mock_restart_server):
    """再起動ツールのテスト"""
    mock_restart_server.return_value = "Restarted successfully"
    result = await handle_call_tool(
        "restart_llm_server", {"port": 8080, "model_name": "new-model", "memory_requirement_gb": 5.0}
    )
    assert result[0].text == "Restarted successfully"
    mock_restart_server.assert_called_once_with(8080, "new-model", 10, 5.0)


@pytest.mark.asyncio
async def test_restart_llm_server_invalid_args():
    """引数バリデーションテスト (restart_llm_server)"""
    with pytest.raises(ValueError, match="Port must be an integer"):
        await handle_call_tool("restart_llm_server", {"port": "8080"})
    with pytest.raises(ValueError, match="model_name must be a string"):
        await handle_call_tool("restart_llm_server", {"port": 8080, "model_name": 123})
    with pytest.raises(ValueError, match="memory_requirement_gb must be a number"):
        await handle_call_tool("restart_llm_server", {"port": 8080, "model_name": "test", "memory_requirement_gb": "8"})


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


@pytest.mark.asyncio
@patch("mcp.server.stdio.stdio_server")
@patch("mcp_mlx_launcher.server.server.run", new_callable=AsyncMock)
@patch("mcp_mlx_launcher.server.process_manager.get_running_servers")
@patch("mcp_mlx_launcher.server.process_manager.shutdown_server")
async def test_run_cleanup_on_exit(mock_shutdown, mock_get_servers, mock_server_run, mock_stdio):
    """サーバー正常終了時にプロセスが自動でクリーンアップされるかのテスト"""
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = (MagicMock(), MagicMock())
    mock_ctx.__aexit__.return_value = None
    mock_stdio.return_value = mock_ctx

    mock_get_servers.return_value = {"8080": {"pid": 111, "model": "test1"}}

    await run()

    mock_get_servers.assert_called_once()
    mock_shutdown.assert_called_once_with(8080)


@pytest.mark.asyncio
@patch("mcp.server.stdio.stdio_server")
@patch("mcp_mlx_launcher.server.server.run", new_callable=AsyncMock)
@patch("mcp_mlx_launcher.server.process_manager.get_running_servers")
@patch("mcp_mlx_launcher.server.process_manager.shutdown_server")
async def test_run_cleanup_on_exception(mock_shutdown, mock_get_servers, mock_server_run, mock_stdio):
    """サーバー異常終了時（例外発生時）にもプロセスが自動でクリーンアップされるかのテスト"""
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = (MagicMock(), MagicMock())
    mock_ctx.__aexit__.return_value = None
    mock_stdio.return_value = mock_ctx

    mock_server_run.side_effect = Exception("Server crash")
    mock_get_servers.return_value = {"8080": {"pid": 111, "model": "test1"}, "8081": {"pid": 222, "model": "test2"}}

    with pytest.raises(Exception, match="Server crash"):
        await run()

    mock_get_servers.assert_called_once()
    assert mock_shutdown.call_count == 2
    mock_shutdown.assert_any_call(8080)
    mock_shutdown.assert_any_call(8081)