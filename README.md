# LocalFileAgent

A command-line tool that scans local files and directories, then either summarises their contents or lets you chat with them interactively — all powered by a locally-running [Ollama](https://ollama.com) model. No data leaves your machine.

## Features

### GUI (PySide6)
- **Unified workspace** — context sidebar, streaming chat, and on-demand summarization in one view
- **Context sidebar** — always shows loaded files with status badges (✓ loaded / ⚠ skipped / deleted), per-file token estimates, and a live token progress bar
- **Streaming responses** — model replies render token-by-token with a blinking cursor; UI stays responsive throughout
- **Per-file summarization** — click any file's button in the summarize strip to generate a summary; copy all as Markdown or save to a `.md` file
- **Session history** — sessions auto-save after every reply; browse, load, or delete past sessions via the Sessions panel
- **Drag-and-drop** — drop files or folders directly onto the sidebar; also use `+ Files` / `+ Folder` buttons

### CLI
- **Summarise mode** — generates a concise 3-5 sentence summary for every matched file
- **Chat mode** — loads file contents into context and opens a REPL for Q&A
- Supports a wide range of **text formats**: `.py`, `.js`, `.ts`, `.md`, `.json`, `.yaml`, `.sql`, `.html`, `.csv`, and more
- Supports **binary document formats**: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.pptx`, `.ppt`
- Optional **recursive** directory scanning
- Output summaries to a plain text or **Markdown** file
- Filter by **file extension**
- Works with any Ollama-compatible model (default: `mistral`)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- At least one model pulled, e.g. `ollama pull mistral`
- `PySide6` for the GUI: `pip install PySide6`

### Optional dependencies (for binary file formats)

| Format | Package |
|--------|---------|
| `.pdf` | `pip install pypdf` |
| `.docx` | `pip install python-docx` |
| `.xlsx` | `pip install openpyxl` |
| `.xls` | `pip install xlrd` |
| `.pptx` | `pip install python-pptx` |
| `.doc` / `.ppt` | `pip install pywin32` (Windows + Microsoft Office required) |

## Usage

```bash
# Launch the GUI
python gui.py

# Summarise all supported files in a directory
python LocalfileAgent.py /path/to/directory

# Summarise specific files
python LocalfileAgent.py file1.txt file2.py

# Save summaries to a Markdown file
python LocalfileAgent.py /path/to/directory --output summaries.md

# Only scan .py and .md files, recursively
python LocalfileAgent.py /path/to/directory --ext .py .md --recursive

# Chat mode — ask questions about the files interactively
python LocalfileAgent.py /path/to/directory --chat

# Use a different model
python LocalfileAgent.py /path/to/directory --chat --model gemma3
```

## CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `paths` | | One or more files or directories to scan (required) |
| `--chat` | `-c` | Interactive Q&A mode instead of summarisation |
| `--model` | `-m` | Ollama model to use (default: `mistral`) |
| `--output` | `-o` | Write summaries to a file; `.md` extension produces Markdown |
| `--ext` | | Limit to specific extensions, e.g. `--ext .py .md` |
| `--recursive` | `-r` | Recurse into subdirectories |
| `--no-check` | | Skip the Ollama availability check on startup |

## Chat Mode Commands

Once in chat mode, the following commands are available:

| Command | Action |
|---------|--------|
| `/list` | Show all loaded files |
| `/clear` | Reset conversation history |
| `/help` | Show command reference |
| `/quit` | Exit |

## Privacy

All processing happens locally. Files are read from disk and sent only to the Ollama API running on `localhost:11434`. Nothing is transmitted to external servers.

## Limits

| Setting | Default |
|---------|---------|
| Max file size (text) | 200 KB |
| Max extracted text (binary) | 400 KB |
| Max files in chat context | 20 |

## License

MIT
