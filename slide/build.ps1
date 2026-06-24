$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$localTectonic = Join-Path $projectRoot "tools\tectonic\tectonic.exe"
if (Test-Path $localTectonic) {
    & $localTectonic main.tex
    exit $LASTEXITCODE
}

if (Get-Command latexmk -ErrorAction SilentlyContinue) {
    latexmk -pdf main.tex
    exit $LASTEXITCODE
}

if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
    pdflatex main.tex
    pdflatex main.tex
    exit $LASTEXITCODE
}

if (Get-Command tectonic -ErrorAction SilentlyContinue) {
    tectonic main.tex
    exit $LASTEXITCODE
}

Write-Error "No LaTeX engine found. Install TeX Live/MiKTeX, install Tectonic, or place tectonic.exe at .\tools\tectonic\tectonic.exe."
