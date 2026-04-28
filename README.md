# mcp-mlx-launcher

An MCP (Model Context Protocol) server designed to autonomously manage, launch, and shutdown local `mlx-lm` instances on Apple Silicon (Mac) environments.

This tool empowers AI agents (like Cline, Claude Desktop, etc.) to start local LLM servers on demand, check their status, and gracefully shut them down when no longer needed, saving system resources.

## Features

- **Launch Local LLMs**: Start an `mlx-lm` server with any supported model in the background.
- **Status Check**: Verify if a specific port is currently active and listening.
- **Graceful Shutdown**: Safely terminate the LLM process running on a specific port.
- **Apple Silicon Optimized**: Built specifically to manage MLX-based local models.

## Prerequisites

- macOS (Apple Silicon M1/M2/M3/M4)
- Python 3.10 or higher
- `mlx-lm` installed in your environment (`pip install mlx-lm`)

## Installation

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/mcp-mlx-launcher.git](https://github.com/YOUR_USERNAME/mcp-mlx-launcher.git)
cd mcp-mlx-launcher

# Install dependencies
pip install -e .
```

## Usage (MCP Configuration)

To use this server with your MCP client (e.g., Claude Desktop or Cline), add the following to your MCP configuration file:

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

## Available Tools

Once connected, the MCP server provides the following tools to the AI agent:

1. `check_llm_status(port: int)`: Returns `true` if a server is running on the specified port.
2. `launch_llm_server(model_name: str, port: int)`: Launches an `mlx_lm.server` instance in the background.
3. `shutdown_llm_server(port: int)`: Gracefully terminates the running LLM server on the given port.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
