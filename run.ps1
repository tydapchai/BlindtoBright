$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
& "$ScriptDir\.venv\Scripts\python.exe" "$ScriptDir\app.py" $args
