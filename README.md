# LocalFileAgent

A command-line tool that scans local files and directories, then either summarises their contents or lets you chat with them interactively — all powered by a locally-running [Ollama](https://ollama.com) model. No data leaves your machine.

Chat mode uses **RAG** (retrieval-augmented generation): your files are chunked and embedded once, and each question retrieves only the most relevant passages instead of stuffing every file into the prompt. The result is faster, more reliable answers — and it scales to far more content than a single context window. Pass `--no-rag` to fall back to full-context stuffing.

## Features

### GUI (PySide6)
- **Unified workspace** — context sidebar, streaming chat, and on-demand summarization in one view
- **RAG-powered chat** — files are indexed once (with a pickable embedding model); each question retrieves the most relevant chunks
- **Context sidebar** — always shows loaded files with status badges (✓ loaded / ⚠ skipped / deleted), per-file token estimates, a live token progress bar, and **model + embed-model pickers**
- **Streaming responses** — model replies render token-by-token with a blinking cursor; UI stays responsive throughout
- **Busy indicator** — an animated status-bar bar with a live elapsed-time timer shows when the app is working (Indexing… / Thinking… / Responding… / Summarizing…) so it never looks frozen
- **Per-file summarization** — click any file's button in the summarize strip to generate a summary; copy all as Markdown or save to a `.md` file
- **Session history** — sessions auto-save after every reply; browse, load, or delete past sessions via the Sessions panel
- **Drag-and-drop** — drop files or folders directly onto the sidebar; also use `+ Files` / `+ Folder` buttons

### CLI
- **Summarise mode** — generates a concise 3-5 sentence summary for every matched file
- **Chat mode** — RAG-based Q&A REPL: embeds your files once, then retrieves the top-k relevant chunks per question (`--no-rag` for full-context stuffing)
- Supports a wide range of **text formats**: `.py`, `.js`, `.ts`, `.md`, `.json`, `.yaml`, `.sql`, `.html`, `.csv`, and more
- Supports **binary document formats**: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.pptx`, `.ppt`
- Optional **recursive** directory scanning
- Output summaries to a plain text or **Markdown** file
- Filter by **file extension**
- Works with any Ollama-compatible model (default: `mistral`)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- A chat model pulled, e.g. `ollama pull mistral`
- An embeddings model for RAG chat, e.g. `ollama pull nomic-embed-text`
- Runtime deps: `pip install -r requirements.txt` (numpy, for the RAG index)
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

# Chat mode — RAG Q&A over the files interactively
python LocalfileAgent.py /path/to/directory --chat

# Pick the embedding model and how many chunks to retrieve per question
python LocalfileAgent.py /path/to/directory --chat --embed-model nomic-embed-text --top-k 8

# Disable RAG — load full file contents into context instead
python LocalfileAgent.py /path/to/directory --chat --no-rag

# Use a different chat model
python LocalfileAgent.py /path/to/directory --chat --model gemma3
```

## CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `paths` | | One or more files or directories to scan (required) |
| `--chat` | `-c` | Interactive Q&A mode instead of summarisation |
| `--model` | `-m` | Ollama chat model to use (default: `mistral`) |
| `--embed-model` | | Embedding model for RAG (default: `nomic-embed-text`) |
| `--top-k` | | (chat) Number of context chunks retrieved per question (default: 5) |
| `--no-rag` | | (chat) Disable RAG; load full file contents into context instead |
| `--output` | `-o` | Write summaries to a file; `.md` extension produces Markdown |
| `--ext` | | Limit to specific extensions, e.g. `--ext .py .md` |
| `--recursive` | `-r` | Recurse into subdirectories |
| `--no-check` | | Skip the Ollama availability check on startup |

## Chat Mode Commands

Once in chat mode, the following commands are available:

| Command | Action |
|---------|--------|
| `/clear` | Reset conversation history |
| `/quit` (`/exit`, `/q`) | Exit |

## Privacy

All processing happens locally. Files are read from disk and sent only to the Ollama API running on `localhost:11434`. Nothing is transmitted to external servers.

## Limits

| Setting | Default |
|---------|---------|
| Max file size (text) | 200 KB |
| Max extracted text (binary) | 400 KB |
| Max files in chat context | 20 |
| RAG chunk size / overlap | 900 / 150 chars |
| RAG chunks retrieved per question (`--top-k`) | 5 |

The embedding index is cached under `~/.localfileagent/index/`, keyed by each file's path, size, modification time, and embedding model — so unchanged files are embedded only once.

## License

MIT
