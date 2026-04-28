import os
import json
import subprocess
import psutil
from pathlib import Path


class MlxProcessManager:
    """
    mlx_lm.server のプロセスを管理し、起動・停止・死活監視を行うクラス。
    稼働中のプロセス情報は指定されたディレクトリの JSON ファイルに保存される。
    """

    def __init__(self, state_dir: str = "~/.mcp-mlx-launcher"):
        self.state_dir = Path(state_dir).expanduser()
        self.state_file = self.state_dir / "state.json"
        self._ensure_state_dir()

    def _ensure_state_dir(self):
        """状態保存用のディレクトリとファイルを準備する"""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self._save_state({})

    def _load_state(self) -> dict:
        """状態ファイルから ポート -> PID のマッピングを読み込む"""
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_state(self, state: dict):
        """ポート -> PID のマッピングを状態ファイルに保存する"""
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
            # 権限不足で全ての接続が見れない場合でも、ポートバインドを試みることで確認可能ですが、
            # 通常のユーザーポート(>1024)であれば基本的に net_connections で確認可能です。
            pass
        return False

    def launch_server(self, model_name: str, port: int) -> str:
        """mlx_lm.server をバックグラウンドで起動する"""
        if self.is_port_in_use(port):
            return f"Error: Port {port} is already in use. Please choose another port or shut down the existing process."

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
            # バックグラウンドプロセスとして起動 (標準出力・エラーは/dev/nullへ捨て、セッションを切り離す)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # 状態を記録
            state = self._load_state()
            state[str(port)] = process.pid
            self._save_state(state)

            return f"Successfully launched mlx_lm.server for model '{model_name}' on port {port} (PID: {process.pid})."

        except Exception as e:
            return f"Error launching process: {str(e)}"

    def shutdown_server(self, port: int) -> str:
        """指定されたポートで稼働しているサーバープロセスを終了させる"""
        state = self._load_state()
        port_str = str(port)

        if port_str not in state:
            return f"Error: No running server found on port {port} in the local state."

        pid = state[port_str]

        try:
            proc = psutil.Process(pid)
            proc.terminate()  # SIGTERM を送信
            proc.wait(timeout=5)  # 終了まで最大5秒待機
        except psutil.NoSuchProcess:
            # プロセスが既に存在しない場合は状態ファイルのみクリーンアップする
            pass
        except psutil.TimeoutExpired:
            # 5秒待っても終了しない場合は強制終了 (SIGKILL)
            proc.kill()
        except Exception as e:
            return f"Error shutting down process PID {pid}: {str(e)}"

        # 正常に終了（または既に終了していた）場合は状態ファイルから削除
        del state[port_str]
        self._save_state(state)

        return f"Successfully shut down server on port {port} (PID: {pid})."
        