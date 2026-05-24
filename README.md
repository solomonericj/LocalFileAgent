# LocalFileAgent

A tool that scans local files and directories, then either summarises their contents or lets you chat with them interactively — all powered by a locally-running [Ollama](https://ollama.com) model. No data leaves your machine.

Available as both a **desktop GUI** (`gui.py`) and a **command-line tool** (`LocalfileAgent.py`).

## Features

- **Summarise mode** — generates a concise 3-5 sentence summary for every matched file
- **Chat mode** — loads file contents into context for interactive Q&A
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
- **GUI only:** `pip install PySide6`

### Optional dependencies (for binary file formats)

| Format | Package |
|--------|---------|
| `.pdf` | `pip install pypdf` |
| `.docx` | `pip install python-docx` |
| `.xlsx` | `pip install openpyxl` |
| `.xls` | `pip install xlrd` |
| `.pptx` | `pip install python-pptx` |
| `.doc` / `.ppt` | `pip install pywin32` (Windows + Microsoft Office required) |

## GUI

Launch the desktop interface:

```bash
pip install PySide6
python gui.py
```

The GUI provides:

- **File panel** — add individual files or entire folders via the OS file picker; remove or clear selections at any time
- **Summarize tab** — run summarization with a progress bar, view results inline, and save to `.md` or `.txt`
- **Chat tab** — load files into context and have a multi-turn conversation; supports history clearing
- **Options bar** — model dropdown (auto-populated from Ollama), recursive toggle, extension filter
- All Ollama calls run in background threads so the window stays responsive

## CLI

```bash
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

### CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `paths` | | One or more files or directories to scan (required) |
| `--chat` | `-c` | Interactive Q&A mode instead of summarisation |
| `--model` | `-m` | Ollama model to use (default: `mistral`) |
| `--output` | `-o` | Write summaries to a file; `.md` extension produces Markdown |
| `--ext` | | Limit to specific extensions, e.g. `--ext .py .md` |
| `--recursive` | `-r` | Recurse into subdirectories |
| `--no-check` | | Skip the Ollama availability check on startup |

### Chat Mode Commands

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
