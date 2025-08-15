# Repo cleanup (non-destructive): format, sort imports, lint
param(
  [string]$Path = "."
)

Write-Host "Running cleanup in" (Resolve-Path $Path)

# Prefer local venv if present
$python = "python"
if (Test-Path ".venv/Scripts/python.exe") { $python = ".venv/Scripts/python.exe" }

# Install tools if missing
$tools = @("ruff", "black", "isort")
foreach ($t in $tools) {
  if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
    Write-Host "Installing $t..."
    & $python -m pip install $t
  }
}

# Run tools
ruff check $Path --fix
ruff format $Path
isort $Path
black $Path

Write-Host "Cleanup complete."
