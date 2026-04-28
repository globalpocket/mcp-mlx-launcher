# mcp-mlx-launcher

Apple Silicon (Mac) 環境でローカルの `mlx-lm` インスタンスを自律的に管理、起動、停止するように設計された MCP (Model Context Protocol) サーバーです。

このツールを使用すると、AI エージェント (Cline、Claude Desktop など) が必要に応じてローカル LLM サーバーを起動し、ステータスを確認し、不要になったときに正常に停止してシステムリソースを節約できるようになります。

## 機能

- **ローカル LLM の起動**: 背景でサポートされている任意のモデルを使用して `mlx-lm` サーバーを起動します。
- **ステータス確認**: 特定のポートが現在アクティブでリスニング中かどうかを確認します。
- **正常なシャットダウン**: 特定のポートで実行されている LLM プロセスを安全に終了します。
- **Apple Silicon 最適化**: MLX ベースのローカルモデルを管理するために特別に構築されています。

## 前提条件

- macOS (Apple Silicon M1/M2/M3/M4)
- Python 3.10 以上
- 環境にインストールされた `mlx-lm` (`pip install mlx-lm`)

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/mcp-mlx-launcher.git
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

1. `check_llm_status(port: int)`: 指定されたポートでサーバーが実行されている場合は `true` を返します。
2. `launch_llm_server(model_name: str, port: int)`: 背景で `mlx_lm.server` インスタンスを起動します。
3. `shutdown_llm_server(port: int)`: 指定されたポートで実行中の LLM サーバーを正常に終了します。

## ライセンス

このプロジェクトは MIT ライセンスの下でライセンスされています。詳細は [LICENSE](LICENSE) ファイルを参照してください。
