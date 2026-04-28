import os
import json
import subprocess
import psutil
import time
from pathlib import Path
from filelock import FileLock


class MlxProcessManager:
    """
    mlx_lm.server のプロセスを管理し、起動・停止・死活監視を行うクラス。
    ファイルロックと起動後の生存確認により、高い信頼性を確保する。
    """

    def __init__(self, state_dir: str = "~/.mcp-mlx-launcher"):
        self.state_dir = Path(state_dir).expanduser()
        self.state_file = self.state_dir / "state.json"
        self.lock_file = self.state_dir / "state.json.lock"
        self._ensure_state_dir()

    def _ensure_state_dir(self):
        """状態保存用のディレクトリとファイルを準備する"""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self._save_state({})

    def _load_state(self) -> dict:
        """状態ファイルから ポート -> PID のマッピングを読み込む（ロック付き）"""
        with FileLock(self.lock_file):
            try:
                if not self.state_file.exists():
                    return {}
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}

    def _save_state(self, state: dict):
        """ポート -> PID のマッピングを状態ファイルに保存する（ロック付き）"""
        with FileLock(self.lock_file):
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)

    def is_port_in_use(self, port: int) -> bool:
        """指定されたポートが LISTEN 状態（使用中）かどうかを判定する"""
        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr.port == port and conn.status == "LISTEN":
                    return True
            return False
        except psutil.AccessDenied:
            return False

    def launch_server(self, model_name: str, port: int, timeout: int = 10) -> str:
        """mlx_lm.server を起動し、生存とポートの開放を確認する"""
        if self.is_port_in_use(port):
            return f"Error: Port {port} is already in use."

        cmd = [
            "python",
            "-m",
            "mlx_lm.server",
            "--model",
            model_name,
            "--port",
            str(port),
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # 生存確認ループ (ヘルスチェック)
            start_time = time.time()
            is_verified = False
            
            while time.time() - start_time < timeout:
                # 1. プロセスが即座にクラッシュしていないか確認
                poll_result = process.poll()
                if poll_result is not None:
                    return f"Error: Process exited immediately with code {poll_result}. Check if the model name is correct."

                # 2. ポートがリッスン状態になったか確認
                if self.is_port_in_use(port):
                    is_verified = True
                    break
                
                time.sleep(0.5)

            if not is_verified:
                # タイムアウトしたがプロセスは生きている場合、警告付きで記録
                msg_suffix = " (Warning: Port not yet listening, model might be loading)"
            else:
                msg_suffix = ""

            # 状態を記録
            state = self._load_state()
            state[str(port)] = process.pid
            self._save_state(state)

            return f"Successfully launched '{model_name}' on port {port} (PID: {process.pid}){msg_suffix}."

        except Exception as e:
            return f"Error launching process: {str(e)}"

    def shutdown_server(self, port: int) -> str:
        """指定されたポートで稼働しているサーバープロセスを終了させる"""
        state = self._load_state()
        port_str = str(port)

        if port_str not in state:
            return f"Error: No running server found on port {port}."

        pid = state[port_str]

        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            proc.kill()
        except Exception as e:
            return f"Error during shutdown: {str(e)}"

        # クリーンアップ
        state = self._load_state() # 最新の状態を再取得
        if port_str in state:
            del state[port_str]
            self._save_state(state)

        return f"Successfully shut down server on port {port} (PID: {pid})."