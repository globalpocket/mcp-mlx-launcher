# mcp-mlx-launcher

An MCP (Model Context Protocol) server designed to autonomously manage, launch, and shutdown local `mlx-lm` instances on Apple Silicon (Mac) environments.

This tool empowers AI agents (like Cline, Claude Desktop, etc.) to start local LLM servers on demand, check their status, prepare environments, and gracefully shut them down when no longer needed, saving system resources.

## Features

- **System Environment Check**: Verify system memory and architecture (Apple Silicon) to ensure readiness.
- **Model Search & Download**: Search Hugging Face for available MLX models and download them locally to cache before launching.
- **Launch & Manage Local LLMs**: Start, stop, and restart an `mlx-lm` server with any supported model in the background.
- **Status Check**: Verify if a specific port is currently active and listening.
- **Apple Silicon Optimized**: Built specifically to manage MLX-based local models.
- **Auto Cleanup**: Automatically cleans up and shuts down all managed LLM processes when the MCP server disconnects or shuts down, preventing resource leaks.

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

1. `check_system_environment()`: Diagnoses the current system environment, returning available unified memory (GB) and architecture details.
2. `check_llm_status(port: int)`: Returns `true` if a server is currently running on the specified port.
3. `list_running_servers()`: Retrieves a list of all local LLM servers (ports and models) currently running in the background.
4. `search_mlx_models(search_query: str = "", limit: int = 10)`: Searches Hugging Face for available MLX format models and lists their details (like download count and model ID).
5. `download_model(model_name: str)`: Pre-downloads a specified MLX model from Hugging Face and caches it locally. Useful for preparing large models before launching.
6. `launch_llm_server(model_name: str, port: int, memory_requirement_gb: float = 4.0)`: Launches an `mlx_lm.server` instance in the background. Includes an optional memory requirement check to prevent out-of-memory errors.
7. `restart_llm_server(port: int, model_name: str = None, memory_requirement_gb: float = 4.0)`: Gracefully stops the running server on the given port and restarts it. If `model_name` is omitted, it restarts with the currently loaded model.
8. `shutdown_llm_server(port: int)`: Gracefully terminates the running LLM server on the given port.
