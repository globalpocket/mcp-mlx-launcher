import pytest
from unittest.mock import patch, MagicMock
from mcp_mlx_launcher.process_manager import MlxProcessManager


@pytest.fixture
def manager(tmp_path):
    """
    テスト用のフィクスチャ。
    実際のユーザーディレクトリを汚さないよう、pytestの tmp_path を状態保存先として使用。
    """
    return MlxProcessManager(state_dir=str(tmp_path))


def test_is_port_in_use_false(manager):
    with patch("psutil.net_connections", return_value=[]):
        assert manager.is_port_in_use(8080) is False


def test_is_port_in_use_true(manager):
    mock_conn = MagicMock()
    mock_conn.laddr.port = 8080
    mock_conn.status = "LISTEN"
    
    with patch("psutil.net_connections", return_value=[mock_conn]):
        assert manager.is_port_in_use(8080) is True


@patch("subprocess.Popen")
def test_launch_server_success(mock_popen, manager):
    # Popenの戻り値(プロセスオブジェクト)をモック化
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc

    with patch.object(manager, "is_port_in_use", return_value=False):
        result = manager.launch_server("mlx-community/Llama-3-8B-Instruct-4bit", 8080)
        
        assert "Successfully launched" in result
        assert "12345" in result
        
        # 状態ファイルにPIDが保存されているか確認
        state = manager._load_state()
        assert state["8080"] == 12345


def test_launch_server_port_in_use(manager):
    with patch.object(manager, "is_port_in_use", return_value=True):
        result = manager.launch_server("test_model", 8080)
        assert "Error: Port 8080 is already in use" in result


@patch("psutil.Process")
def test_shutdown_server_success(mock_process, manager):
    # テスト用の状態を準備
    manager._save_state({"8080": 12345})
    
    mock_proc = MagicMock()
    mock_process.return_value = mock_proc
    
    result = manager.shutdown_server(8080)
    
    assert "Successfully shut down" in result
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=5)
    
    # 状態ファイルから削除されているか確認
    state = manager._load_state()
    assert "8080" not in state


def test_shutdown_server_not_found(manager):
    manager._save_state({})  # 空の状態
    result = manager.shutdown_server(8080)
    assert "Error: No running server found" in result