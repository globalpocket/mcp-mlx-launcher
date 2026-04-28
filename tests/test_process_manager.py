import pytest
from unittest.mock import patch, MagicMock
import psutil
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


def test_is_port_in_use_access_denied(manager):
    """psutil.AccessDenied が発生した場合は False を返すことの確認"""
    with patch("psutil.net_connections", side_effect=psutil.AccessDenied):
        assert manager.is_port_in_use(8080) is False


@patch("subprocess.Popen")
def test_launch_server_success(mock_popen, manager):
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc

    with patch.object(manager, "is_port_in_use", return_value=False):
        result = manager.launch_server("mlx-community/Llama-3-8B-Instruct-4bit", 8080)
        
        assert "Successfully launched" in result
        assert "12345" in result
        
        state = manager._load_state()
        assert state["8080"] == 12345


def test_launch_server_port_in_use(manager):
    with patch.object(manager, "is_port_in_use", return_value=True):
        result = manager.launch_server("test_model", 8080)
        assert "Error: Port 8080 is already in use" in result


@patch("subprocess.Popen")
def test_launch_server_exception(mock_popen, manager):
    """サブプロセス起動時に予期せぬエラーが発生した場合のテスト"""
    mock_popen.side_effect = Exception("Unexpected failure")
    
    with patch.object(manager, "is_port_in_use", return_value=False):
        result = manager.launch_server("test_model", 8080)
        assert "Error launching process: Unexpected failure" in result


@patch("psutil.Process")
def test_shutdown_server_success(mock_process, manager):
    manager._save_state({"8080": 12345})
    
    mock_proc = MagicMock()
    mock_process.return_value = mock_proc
    
    result = manager.shutdown_server(8080)
    
    assert "Successfully shut down" in result
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=5)
    
    state = manager._load_state()
    assert "8080" not in state


def test_shutdown_server_not_found(manager):
    manager._save_state({})
    result = manager.shutdown_server(8080)
    assert "Error: No running server found" in result


@patch("psutil.Process")
def test_shutdown_server_no_such_process(mock_process, manager):
    """プロセスが既に存在しない(NoSuchProcess)場合のテスト"""
    manager._save_state({"8080": 12345})
    mock_process.side_effect = psutil.NoSuchProcess(12345)
    
    result = manager.shutdown_server(8080)
    assert "Successfully shut down" in result
    
    # 状態ファイルからは削除されていること
    state = manager._load_state()
    assert "8080" not in state


@patch("psutil.Process")
def test_shutdown_server_timeout_expired(mock_process, manager):
    """terminate 後にタイムアウトし、kill が呼ばれる場合のテスト"""
    manager._save_state({"8080": 12345})
    
    mock_proc = MagicMock()
    # wait 時に TimeoutExpired を発生させる
    mock_proc.wait.side_effect = psutil.TimeoutExpired(5, pid=12345)
    mock_process.return_value = mock_proc
    
    result = manager.shutdown_server(8080)
    
    assert "Successfully shut down" in result
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once() # killが呼ばれたことを確認
    
    state = manager._load_state()
    assert "8080" not in state


@patch("psutil.Process")
def test_shutdown_server_general_exception(mock_process, manager):
    """終了処理中に予期せぬエラーが発生した場合のテスト"""
    manager._save_state({"8080": 12345})
    mock_process.side_effect = Exception("Permission denied")
    
    result = manager.shutdown_server(8080)
    assert "Error shutting down process PID 12345: Permission denied" in result
    
    # エラー時は状態ファイルから削除されないこと
    state = manager._load_state()
    assert "8080" in state