# mcp-mlx-launcher

Apple Silicon (Mac) 環境でローカルの `mlx-lm` インスタンスを自律的に管理、起動、停止するように設計された MCP (Model Context Protocol) サーバーです。

このツールを使用すると、AI エージェント (Cline、Claude Desktop など) が必要に応じてローカル LLM サーバーを起動し、ステータスを確認し、環境を準備し、不要になったときに正常に停止してシステムリソースを節約できるようになります。

## 機能

- **システム環境診断**: システムのメモリやアーキテクチャ（Apple Silicon）の情報を確認し、起動の準備が整っているか診断します。
- **モデルの検索とダウンロード**: Hugging Face上で利用可能なMLXモデルを検索し、起動前にローカルに事前ダウンロード（キャッシュ）します。
- **ローカル LLM の起動と管理**: バックグラウンドでサポートされている任意のモデルを使用して `mlx-lm` サーバーの起動、停止、再起動を行います。
- **ステータス確認**: 特定のポートが現在アクティブでリスニング中かどうかを確認します。
- **Apple Silicon 最適化**: MLX ベースのローカルモデルを管理するために特別に構築されています。
- **自動クリーンアップ**: MCP サーバーの切断または終了時に、管理下にあるすべての LLM プロセスを自動的にクリーンアップおよび停止し、リソースのリークを防ぎます。

## 前提条件

- macOS (Apple Silicon M1/M2/M3/M4)
- Python 3.10 以上
- 環境にインストールされた `mlx-lm` (`pip install mlx-lm`)

## インストール

```bash
# リポジトリをクローン
git clone [https://github.com/YOUR_USERNAME/mcp-mlx-launcher.git](https://github.com/YOUR_USERNAME/mcp-mlx-launcher.git)
cd mcp-mlx-launcher

# 依存関係をインストール
pip install -e .
```

## 使用方法 (MCP 設定)

MCP クライアント (Claude Desktop や Cline など) でこのサーバーを使用するには、MCP 設定ファイルに以下を追加します。

```json
{
  "mcpServers": {
    "mcp-mlx-launcher": {
      "command": "python",
      "args": [
        "-m",
        "mcp_mlx_launcher.server"
      ]
    }
  }
}
```

## 利用可能なツール

接続されると、MCP サーバーは AI エージェントに以下のツールを提供します。

1. `check_system_environment()`: 現在のシステム環境（Apple Siliconか、空きメモリが何GBあるかなど）を診断します。
2. `check_llm_status(port: int)`: 指定されたポートでサーバーが実行されている場合は `true` を返します。
3. `list_running_servers()`: 現在バックグラウンドで稼働しているすべてのローカルLLMサーバー（ポート番号とモデル名）の一覧を取得します。
4. `search_mlx_models(search_query: str = "", limit: int = 10)`: Hugging Faceからダウンロード可能なMLXフォーマットのLLMモデルを検索・リストアップします。
5. `download_model(model_name: str)`: Hugging Faceから指定されたMLXモデルを事前にダウンロードし、ローカルにキャッシュします。大きなモデルの起動前の準備に利用します。
6. `launch_llm_server(model_name: str, port: int, memory_requirement_gb: float = 4.0)`: バックグラウンドで `mlx_lm.server` インスタンスを起動します。メモリ不足エラーを防ぐためのオプションのメモリ要件チェックが含まれています。
7. `restart_llm_server(port: int, model_name: str = None, memory_requirement_gb: float = 4.0)`: 指定されたポートで稼働しているサーバーを一度停止し、再起動します。モデル名を省略した場合は現在そのポートで稼働しているモデルをそのまま再起動します。
8. `shutdown_llm_server(port: int)`: 指定されたポートで実行中の LLM サーバーを正常に終了します。
