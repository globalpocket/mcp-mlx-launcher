import os
import sys
import json
import subprocess
import psutil
import time
import platform
from pathlib import Path
from filelock import FileLock
from huggingface_hub import snapshot_download


class MlxProcessManager:
    """
    mlx_lm.server のプロセスを管理し、起動・停止・死活監視を行うクラス。
    ファイルロックと起動後の生存確認により、高い信頼性を確保する。
    """

    def __init__(self, state_dir: str = "~/.local/share/mcp-mlx-launcher"):
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
        """状態ファイルから ポート -> {pid, model} のマッピングを読み込む（ロック付き）"""
        with FileLock(self.lock_file):
            try:
                if not self.state_file.exists():
                    return {}
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    # 後方互換性: 古いフォーマット {"8080": 12345} を新しいフォーマットに変換
                    for k, v in data.items():
                        if isinstance(v, int):
                            data[k] = {"pid": v, "model": "unknown"}
                    return data
            except (json.JSONDecodeError, FileNotFoundError):
                return {}

    def _save_state(self, state: dict):
        """ポート -> {pid, model} のマッピングを状態ファイルに保存する（ロック付き）"""
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

    def is_model_cached(self, model_name: str) -> bool:
        """Hugging Faceのキャッシュディレクトリにモデルが存在するか確認する"""
        cache_base = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")
        cache_dir = Path(cache_base).expanduser()
        
        folder_name = "models--" + model_name.replace("/", "--")
        model_path = cache_dir / folder_name
        return model_path.exists()

    def get_system_info(self) -> dict:
        """現在のシステム状態（メモリ、アーキテクチャなど）を取得する"""
        mem = psutil.virtual_memory()
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "total_memory_gb": round(mem.total / (1024 ** 3), 2),
            "available_memory_gb": round(mem.available / (1024 ** 3), 2),
            "python_version": platform.python_version()
        }

    def get_running_servers(self) -> dict:
        """現在稼働中のサーバー一覧を取得し、死んだプロセスをクリーンアップする"""
        state = self._load_state()
        active_servers = {}
        changed = False

        for port_str, info in list(state.items()):
            pid = info["pid"]
            if psutil.pid_exists(pid):
                active_servers[port_str] = info
            else:
                del state[port_str]
                changed = True
                
        if changed:
            self._save_state(state)
            
        return active_servers

    def download_model(self, model_name: str) -> str:
        """Hugging Faceからモデルを事前にダウンロード（キャッシュ）する"""
        try:
            snapshot_download(repo_id=model_name)
            return f"Successfully downloaded and cached model: {model_name}"
        except Exception as e:
            return f"Error downloading model {model_name}: {str(e)}"

    def launch_server(self, model_name: str, port: int, timeout: int = 10, memory_requirement_gb: float = 4.0) -> str:
        """mlx_lm.server を起動し、生存とポートの開放を確認する"""
        
        if self.is_port_in_use(port):
            return f"Error: Port {port} is already in use."

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        if available_gb < memory_requirement_gb:
            return f"Error: Insufficient memory. Only {available_gb:.2f}GB available, but at least {memory_requirement_gb}GB is requested to launch this model safely."

        is_cached = self.is_model_cached(model_name)

        cmd = [
            sys.executable,
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

            start_time = time.time()
            is_verified = False
            
            while time.time() - start_time < timeout:
                poll_result = process.poll()
                if poll_result is not None:
                    return f"Error: Process exited immediately with code {poll_result}. Check if the model name is correct or if you have enough unified memory."

                if self.is_port_in_use(port):
                    is_verified = True
                    break
                
                time.sleep(0.5)

            if not is_verified:
                if not is_cached:
                    msg_suffix = " (Note: Model is currently being downloaded from Hugging Face in the background. It may take a while before the port becomes active.)"
                else:
                    msg_suffix = " (Warning: Port not yet listening, model might still be loading into memory)"
            else:
                msg_suffix = ""

            state = self._load_state()
            state[str(port)] = {"pid": process.pid, "model": model_name}
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

        pid = state[port_str]["pid"]

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

        state = self._load_state()
        if port_str in state:
            del state[port_str]
            self._save_state(state)

        return f"Successfully shut down server on port {port} (PID: {pid})."

    def restart_server(self, port: int, model_name: str = None, timeout: int = 10, memory_requirement_gb: float = 4.0) -> str:
        """指定されたポートのサーバーを再起動する"""
        state = self._load_state()
        port_str = str(port)

        if port_str not in state:
            return f"Error: No running server found on port {port} to restart."

        # モデル名が指定されていない場合は、現在のモデルを引き継ぐ
        current_model = state[port_str].get("model", "unknown")
        target_model = model_name if model_name else current_model

        if target_model == "unknown":
            return "Error: Cannot determine the current model to restart. Please specify model_name explicitly."

        shutdown_msg = self.shutdown_server(port)
        if "Error" in shutdown_msg:
            return f"Failed to shutdown existing server: {shutdown_msg}"

        # OSがポートを完全に解放するまで少し待機
        time.sleep(1)

        launch_msg = self.launch_server(target_model, port, timeout, memory_requirement_gb)
        return f"{shutdown_msg}\nRestart Result: {launch_msg}"