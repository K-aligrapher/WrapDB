# Starts the WarpDB dashboard (Streamlit) on http://localhost:8501
$root = $PSScriptRoot
& "$root\.venv\Scripts\streamlit.exe" run "$root\python\dashboard\app.py" @args
