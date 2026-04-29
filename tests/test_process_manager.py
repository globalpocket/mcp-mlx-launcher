import pytest
import time
import os
import sys
import psutil
from unittest.mock import patch, MagicMock
from mcp_mlx_launcher.process_manager import MlxProcessManager


@pytest.fixture
def manager(tmp_path):
    """
    テスト用のフィクスチャ。
    実際のユーザーディレクトリを汚さないよう、pytestの tmp_path を状態保存先として使用。
    """
    return MlxProcessManager(state_dir=str(tmp_path))


def test_file_locking_and_migration(manager):
    """ファイルロックの動作と、古いデータフォーマットのマイグレーションテスト"""
    # 古いフォーマット（PIDのみ）で保存
    with open(manager.state_file, "w") as f:
        f.write('{"8080": 12345}')
    
    state = manager._load_state()
    assert state["8080"]["pid"] == 12345
    assert state["8080"]["model"] == "unknown"


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


@patch("psutil.virtual_memory")
def test_get_system_info(mock_vmem, manager):
    """システム情報取得のテスト"""
    mock_mem_obj = MagicMock()
    mock_mem_obj.total = 16 * (1024 ** 3)
    mock_mem_obj.available = 8 * (1024 ** 3)
    mock_vmem.return_value = mock_mem_obj

    info = manager.get_system_info()
    assert info["total_memory_gb"] == 16.0
    assert info["available_memory_gb"] == 8.0
    assert "system" in info
    assert "machine" in info


@patch("mcp_mlx_launcher.process_manager.snapshot_download")
def test_download_model_success(mock_snapshot, manager):
    """モデルダウンロードの成功テスト"""
    result = manager.download_model("test-model")
    assert "Successfully downloaded" in result
    mock_snapshot.assert_called_once_with(repo_id="test-model")


@patch("mcp_mlx_launcher.process_manager.snapshot_download")
def test_download_model_error(mock_snapshot, manager):
    """モデルダウンロードの異常系テスト"""
    mock_snapshot.side_effect = Exception("Network error")
    result = manager.download_model("test-model")
    assert "Error downloading model" in result
    assert "Network error" in result


def test_launch_server_port_in_use(manager):
    with patch.object(manager, "is_port_in_use", return_value=True):
        result = manager.launch_server("test_model", 8080)
        assert "Error: Port 8080 is already in use" in result


@patch("psutil.virtual_memory")
def test_launch_server_insufficient_memory(mock_vmem, manager):
    """メモリ不足時の起動ブロックと、パラメータ変更のテスト"""
    mock_mem_obj = MagicMock()
    mock_mem_obj.available = 2 * (1024 ** 3) # 2GBの空き
    mock_vmem.return_value = mock_mem_obj
    
    with patch.object(manager, "is_port_in_use", return_value=False):
        # デフォルト (4.0GB要求) -> エラー
        result = manager.launch_server("test-model", 8080)
        assert "Insufficient memory" in result
        
        # パラメータ指定 (3.0GB要求) -> それでも2GBしかないのでエラー
        result2 = manager.launch_server("test-model", 8080, memory_requirement_gb=3.0)
        assert "Insufficient memory" in result2


@patch("subprocess.Popen")
@patch("time.sleep", return_value=None)
@patch("psutil.virtual_memory")
@patch.object(MlxProcessManager, "is_model_cached", return_value=True)
def test_launch_server_with_health_check_success(mock_cached, mock_vmem, mock_sleep, mock_popen, manager):
    """起動後、ポートが開放されるまで待機して成功するケースと、sys.executableの検証"""
    mock_mem_obj = MagicMock()
    mock_mem_obj.available = 8 * (1024 ** 3) # 8GBの空き
    mock_vmem.return_value = mock_mem_obj

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None # 生存中
    mock_proc.pid = 9999
    mock_popen.return_value = mock_proc

    # 1回目は未開放、2回目で開放されている状態をシミュレート
    with patch.object(manager, "is_port_in_use", side_effect=[False, True]):
        result = manager.launch_server("test-model", 8080)
        
        assert "Successfully launched" in result
        
        state = manager._load_state()
        assert state["8080"]["pid"] == 9999
        assert state["8080"]["model"] == "test-model"

        mock_popen.assert_called_once()
        args, _ = mock_popen.call_args
        called_cmd = args[0]
        assert called_cmd[0] == sys.executable
        assert called_cmd[1] == "-m"
        assert called_cmd[2] == "mlx_lm.server"


@patch("subprocess.Popen")
@patch("time.sleep", return_value=None)
@patch("psutil.virtual_memory")
def test_launch_server_immediate_crash(mock_vmem, mock_sleep, mock_popen, manager):
    """起動直後にプロセスが終了してしまった場合のエラー検知"""
    mock_mem_obj = MagicMock()
    mock_mem_obj.available = 8 * (1024 ** 3)
    mock_vmem.return_value = mock_mem_obj

    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1 # エラー終了
    mock_popen.return_value = mock_proc

    with patch.object(manager, "is_port_in_use", return_value=False):
        result = manager.launch_server("test-model", 8080)
        assert "Error: Process exited immediately" in result


@patch("subprocess.Popen")
@patch("time.sleep", return_value=None)
@patch("psutil.virtual_memory")
@patch.object(MlxProcessManager, "is_model_cached", return_value=True)
def test_launch_server_timeout_warning(mock_cached, mock_vmem, mock_sleep, mock_popen, manager):
    """生存はしているがポートがなかなか開かない場合のタイムアウト警告"""
    mock_mem_obj = MagicMock()
    mock_mem_obj.available = 8 * (1024 ** 3)
    mock_vmem.return_value = mock_mem_obj

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 9999
    mock_popen.return_value = mock_proc

    with patch.object(manager, "is_port_in_use", return_value=False):
        with patch("time.time", side_effect=[0, 11]):
            result = manager.launch_server("test-model", 8080)
            assert "Warning: Port not yet listening" in result


@patch("subprocess.Popen")
@patch("time.sleep", return_value=None)
@patch("psutil.virtual_memory")
@patch.object(MlxProcessManager, "is_model_cached", return_value=False)
def test_launch_server_download_warning(mock_cached, mock_vmem, mock_sleep, mock_popen, manager):
    """未キャッシュ（ダウンロード中）の場合のメッセージテスト"""
    mock_mem_obj = MagicMock()
    mock_mem_obj.available = 8 * (1024 ** 3)
    mock_vmem.return_value = mock_mem_obj

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 9999
    mock_popen.return_value = mock_proc

    with patch.object(manager, "is_port_in_use", return_value=False):
        with patch("time.time", side_effect=[0, 11]):
            result = manager.launch_server("test-model", 8080)
            assert "being downloaded from Hugging Face" in result


@patch("subprocess.Popen")
@patch("psutil.virtual_memory")
def test_launch_server_exception(mock_vmem, mock_popen, manager):
    """サブプロセス起動時に予期せぬエラーが発生した場合のテスト"""
    mock_mem_obj = MagicMock()
    mock_mem_obj.available = 8 * (1024 ** 3)
    mock_vmem.return_value = mock_mem_obj

    mock_popen.side_effect = Exception("Unexpected failure")
    
    with patch.object(manager, "is_port_in_use", return_value=False):
        result = manager.launch_server("test_model", 8080)
        assert "Error launching process: Unexpected failure" in result


@patch("pathlib.Path.exists")
def test_is_model_cached_env_var(mock_exists, manager):
    """環境変数 HF_HOME を尊重するかテスト"""
    mock_exists.return_value = True
    with patch.dict(os.environ, {"HF_HOME": "/custom/hf/cache"}):
        assert manager.is_model_cached("test/model") is True


@patch("psutil.Process")
def test_shutdown_server_success(mock_process, manager):
    """正常なシャットダウンテスト"""
    manager._save_state({"8080": {"pid": 12345, "model": "test"}})
    
    mock_proc = MagicMock()
    mock_process.return_value = mock_proc
    
    result = manager.shutdown_server(8080)
    
    assert "Successfully shut down" in result
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=5)
    
    state = manager._load_state()
    assert "8080" not in state


def test_shutdown_server_not_found(manager):
    """存在しないポートの停止テスト"""
    manager._save_state({})
    result = manager.shutdown_server(8080)
    assert "Error: No running server found" in result


@patch("psutil.Process")
def test_shutdown_server_no_such_process(mock_process, manager):
    """プロセスが既にOS上に存在しない場合のクリーンアップテスト"""
    manager._save_state({"8080": {"pid": 12345, "model": "test"}})
    mock_process.side_effect = psutil.NoSuchProcess(12345)
    
    result = manager.shutdown_server(8080)
    assert "Successfully shut down" in result
    
    state = manager._load_state()
    assert "8080" not in state


@patch("psutil.Process")
def test_shutdown_server_timeout_expired(mock_process, manager):
    """terminate 後にタイムアウトし、kill が呼ばれる場合のテスト"""
    manager._save_state({"8080": {"pid": 12345, "model": "test"}})
    
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = psutil.TimeoutExpired(5, pid=12345)
    mock_process.return_value = mock_proc
    
    result = manager.shutdown_server(8080)
    
    assert "Successfully shut down" in result
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()
    
    state = manager._load_state()
    assert "8080" not in state


@patch("psutil.Process")
def test_shutdown_server_general_exception(mock_process, manager):
    """終了処理中に予期せぬエラーが発生した場合のテスト"""
    manager._save_state({"8080": {"pid": 12345, "model": "test"}})
    mock_process.side_effect = Exception("Permission denied")
    
    result = manager.shutdown_server(8080)
    assert "Error during shutdown: Permission denied" in result
    
    state = manager._load_state()
    assert "8080" in state


@patch("psutil.pid_exists")
def test_get_running_servers(mock_pid_exists, manager):
    """稼働中のサーバー一覧取得と、死んだプロセスの自動クリーンアップテスト"""
    manager._save_state({
        "8080": {"pid": 100, "model": "model-A"}, # 生きてる
        "8081": {"pid": 101, "model": "model-B"}, # 死んでる
    })
    
    mock_pid_exists.side_effect = lambda pid: pid == 100
    
    servers = manager.get_running_servers()
    
    assert "8080" in servers
    assert "8081" not in servers
    assert len(servers) == 1
    
    state = manager._load_state()
    assert "8081" not in state


@patch.object(MlxProcessManager, "shutdown_server")
@patch.object(MlxProcessManager, "launch_server")
@patch("time.sleep", return_value=None)
def test_restart_server_success(mock_sleep, mock_launch, mock_shutdown, manager):
    """再起動テスト（モデル名省略時、同じモデルを引き継ぐか）"""
    manager._save_state({"8080": {"pid": 12345, "model": "old-model"}})
    mock_shutdown.return_value = "Shut down OK"
    mock_launch.return_value = "Launched OK"

    result = manager.restart_server(8080)
    
    assert "Shut down OK" in result
    assert "Launched OK" in result
    mock_shutdown.assert_called_once_with(8080)
    mock_launch.assert_called_once_with("old-model", 8080, 10, 4.0)


@patch.object(MlxProcessManager, "shutdown_server")
@patch.object(MlxProcessManager, "launch_server")
@patch("time.sleep", return_value=None)
def test_restart_server_with_new_model(mock_sleep, mock_launch, mock_shutdown, manager):
    """再起動テスト（新しいモデル名指定時）"""
    manager._save_state({"8080": {"pid": 12345, "model": "old-model"}})
    mock_shutdown.return_value = "Shut down OK"
    mock_launch.return_value = "Launched OK"

    result = manager.restart_server(8080, model_name="new-model", memory_requirement_gb=5.0)
    
    mock_shutdown.assert_called_once_with(8080)
    mock_launch.assert_called_once_with("new-model", 8080, 10, 5.0)


def test_restart_server_not_found(manager):
    """稼働していないポートの再起動テスト"""
    result = manager.restart_server(8080)
    assert "Error: No running server found" in result