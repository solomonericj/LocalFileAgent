# LocalFileAgent

A command-line tool that scans local files and directories, then either summarises their contents or lets you chat with them interactively — all powered by a locally-running [Ollama](https://ollama.com) model. No data leaves your machine.

Chat is a **general assistant**: it answers any question from the model's own knowledge and works with **zero files loaded**. Point it at files and they become extra context — the assistant draws on them and cites which file it used when relevant, but it won't refuse a question just because the answer isn't in your files.

When files _are_ loaded, chat uses **RAG** (retrieval-augmented generation): your files are chunked and embedded once, and each question retrieves only the most relevant passages instead of stuffing every file into the prompt. The result is faster, more reliable answers — and it scales to far more content than a single context window. Pass `--no-rag` to fall back to full-context stuffing.

Before answering each message, the assistant rates its own confidence in understanding the request; if it's unsure it asks a focused **clarifying question** first (up to three per turn) instead of guessing.

## Features

### GUI (PySide6)
- **Unified workspace** — context sidebar, streaming chat, and on-demand summarization in one view
- **General-assistant chat** — chat works with **no files loaded** (answers from the model's own knowledge); load files anytime to add grounded context
- **Clarifying questions** — when the model is unsure what you mean, it asks a focused question inline before answering (up to three per turn)
- **RAG-powered chat** — files are indexed once (with a pickable embedding model); each question retrieves the most relevant chunks
- **Context sidebar** — always shows loaded files with status badges (✓ loaded / ⚠ skipped / deleted), per-file token estimates, a live token progress bar, and **model + embed-model pickers**
- **Streaming responses** — model replies render token-by-token with a blinking cursor; UI stays responsive throughout
- **Busy indicator** — an animated status-bar bar with a live elapsed-time timer shows when the app is working (Indexing… / Thinking… / Responding… / Summarizing…) so it never looks frozen
- **Per-file summarization** — click any file's button in the summarize strip to generate a summary; copy all as Markdown or save to a `.md` file
- **Session history** — sessions auto-save after every reply; browse, load, or delete past sessions via the Sessions panel
- **Drag-and-drop** — drop files or folders directly onto the sidebar; also use `+ Files` / `+ Folder` buttons

### CLI
- **Summarise mode** — generates a concise 3-5 sentence summary for every matched file
- **Chat mode** — a general-assistant REPL that works with or without files. With no paths (`localfileagent --chat`) it's a plain Q&A assistant; with files it embeds them once and retrieves the top-k relevant chunks per question (`--no-rag` for full-context stuffing)
- **Clarifying questions** — the assistant probes its own confidence before each answer and asks a focused question when it's unsure (up to three per turn)
- Supports a wide range of **text formats**: code (`.py`, `.js`, `.ts`, `.go`, `.rs`, `.dart`, `.scala`, `.lua`, `.ex`, `.r`, `.zig`, …), config/IaC (`.json`, `.yaml`, `.toml`, `.ini`, `.cfg`, `.env`, `.tf`, `.tfvars`, `.hcl`, `.ps1`, `.bat`), web (`.html`, `.css`, `.vue`, `.svelte`), schema/IDL (`.graphql`, `.proto`), and more
- Supports **binary & packaged formats**: documents (`.pdf`, `.docx`, `.xlsx`, `.xls`, `.pptx`, `.ppt`, `.odt`, `.ods`, `.odp`), notebooks/email (`.ipynb`, `.eml`), and archives (`.zip`, `.7z`, `.tar`, `.tgz`)
- Optional **recursive** directory scanning
- Output summaries to a plain text or **Markdown** file
- Filter by **file extension**
- Works with any Ollama-compatible model (default: `mistral`)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- A chat model pulled, e.g. `ollama pull mistral`
- An embeddings model for RAG chat, e.g. `ollama pull nomic-embed-text`
- Runtime deps (numpy + PySide6): installed automatically by `pip install -e .` (see [Install](#install)), or via `pip install -r requirements.txt`

### Optional dependencies (for binary file formats)

Notebooks (`.ipynb`), email (`.eml`), and `.zip` / `.tar` / `.tgz` archives are parsed with the **Python standard library** — no extra packages needed. The formats below need an optional dependency:

| Format | Package |
|--------|---------|
| `.pdf` | `pip install pypdf` |
| `.docx` | `pip install python-docx` |
| `.xlsx` | `pip install openpyxl` |
| `.xls` | `pip install xlrd` |
| `.pptx` | `pip install python-pptx` |
| `.odt` / `.ods` / `.odp` | `pip install odfpy` |
| `.7z` | `pip install py7zr` |
| `.doc` / `.ppt` | `pip install pywin32` (Windows + Microsoft Office required) |

## Install

Install once as an editable package. This pulls in the runtime dependencies **and** puts a `localfileagent` command on your PATH, so you can run it from any directory:

```bash
pip install -e .
```

Then run it anywhere:

```bash
localfileagent --gui                 # graphical interface
localfileagent /path/to/dir --chat   # CLI chat
```

Optional extras — install only what you need:

```bash
pip install -e ".[parsers]"   # all cross-platform binary parsers: .pdf .docx .xlsx .xls .pptx .odt/.ods/.odp .7z
pip install -e ".[pdf]"       # a single format (also: docx, xlsx, xls, pptx, odf, 7z)
pip install -e ".[legacy]"    # .doc / .ppt via Office COM (Windows only)
pip install -e ".[dev]"       # pytest + pytest-qt for the test suite
```

> **`localfileagent: command not found`?** pip installed the script into your Python *scripts* directory, which isn't on your PATH. Print the directory with `python -c "import sysconfig; print(sysconfig.get_path('scripts'))"` and add it to PATH (on Windows it's typically `…\PythonXY\Scripts`, or `…\AppData\Roaming\Python\PythonXY\Scripts` for `--user` installs). You can always skip installing and run `python LocalfileAgent.py …` / `python gui.py` directly.

## Usage

```bash
# Launch the GUI
localfileagent --gui

# Summarise all supported files in a directory
localfileagent /path/to/directory

# Summarise specific files
localfileagent file1.txt file2.py

# Save summaries to a Markdown file
localfileagent /path/to/directory --output summaries.md

# Only scan .py and .md files, recursively
localfileagent /path/to/directory --ext .py .md --recursive

# General-assistant chat — no files, answers from the model's own knowledge
localfileagent --chat

# Chat mode — RAG Q&A over the files interactively
localfileagent /path/to/directory --chat

# Pick the embedding model and how many chunks to retrieve per question
localfileagent /path/to/directory --chat --embed-model nomic-embed-text --top-k 8

# Disable RAG — load full file contents into context instead
localfileagent /path/to/directory --chat --no-rag

# Use a different chat model
localfileagent /path/to/directory --chat --model gemma3
```

> Not installed? Replace `localfileagent` with `python LocalfileAgent.py` (and `localfileagent --gui` with `python gui.py`).

## CLI Options

| Flag | Short | Description |
|------|-------|-------------|
| `paths` | | One or more files or directories to scan (required for summarise; optional for `--chat` / `--gui`) |
| `--gui` | | Launch the graphical interface instead of the CLI |
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

## Security

LocalFileAgent is a local tool you run on your own machine, against files you choose. Be aware of what that involves:

- **It reads arbitrary local files.** Whatever paths you point it at are read from disk and their contents sent to your local Ollama model. Only scan files you trust and intend to share with the model.
- **It runs third-party document parsers.** Opening `.pdf`, `.docx`, `.xlsx`, `.xls`, `.pptx`, `.odt`/`.ods`/`.odp`, and `.7z` files invokes the corresponding parsing libraries (`pypdf`, `python-docx`, `openpyxl`, `xlrd`, `python-pptx`, `odfpy`, `py7zr`). A maliciously crafted document could exploit a bug in one of these parsers, so keep them updated and treat untrusted documents with caution.
- **Archives are read, not extracted to disk.** `.zip`, `.7z`, `.tar`, and `.tgz` contents are streamed in memory and only their text-extension entries are read, but pointing the tool at an untrusted archive still feeds attacker-controlled content to the model — treat it like any other untrusted input.
- **`.doc` / `.ppt` launch Microsoft Office via COM.** On Windows, legacy formats are extracted by automating Word/PowerPoint through `pywin32`. This starts real Office processes to open the file — do not point it at untrusted `.doc`/`.ppt` files, since opening them carries the same risks as double-clicking them (e.g. macros).

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
