# Open WebUI Utils

A collection of utilities for [Open WebUI](https://github.com/open-webui/open-webui)

## Native Tool Calling Pipe

A powerful pipe implementation that enables seamless OpenAI API-native tool calling with streaming and multi-call support for Open WebUI.

### Features

- 🔄 Native OpenAI API tool calling format
- ⚡ Streaming support for real-time responses
- 🛠️ Multi-call support for back-and-forth between tools and the assistant
- 🔌 Seamless integration with Open WebUI
- 🎯 Compatible with any OpenAI-compatible endpoint (OpenAI, Ollama, Openrouter,...)

### Demo

https://github.com/user-attachments/assets/f7a07c2f-0a7a-4531-ba40-cc429ca60357

### Requirements

- Open WebUI version 0.5.0 or higher
- OpenAI API key

### Installation

- [Open the function on the community page](https://openwebui.com/f/marcelsamyn/native_tool_calling_pipe)
- Click **Get** and add it to your Open WebUI instance
- Configure the Valves
- In a new conversation, choose one of the new models prefixed with `native-tool/`

### Usage

When you choose one of the models prefixed with `native-tool/`, interaction with the LLM will happen through this pipe and the built-in function calling will be taken over.

### Configuration

The pipe accepts the following configuration options:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: OpenAI API base URL (default: "https://api.openai.com/v1")
- `MODEL_IDS`: List of model IDs to enable (default: ["gpt-4o-mini"])