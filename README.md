# DD2470 Music Informatics Project
Dataset: https://github.com/mdeff/fma

Manage dependencies and run scripts using uv

Setup uv for MacOs / Linux
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Setup uv for Windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then simply do uv run <script> to run a script, and uv add <package> to install a package.
uv automatically manages your .venv for you.


