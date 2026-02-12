param(
    [string]$Command
)

$PythonStr = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonStr = "py"
}
elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonStr = "python3"
}

if ($Command -eq "install") {
    & $PythonStr -m pip install -r requirements.txt
}
elseif ($Command -eq "sim") {
    & $PythonStr simulations/src/run_nested_hs_suite.py --regime all --profile full --seed 42
    & $PythonStr simulations/src/plot_nested_hs_suite.py
    & $PythonStr simulations/src/validate_nested_hs_claims.py
}
elseif ($Command -eq "sim-qutrit") {
    & $PythonStr simulations/src/run_nested_hs_suite.py --regime cg --profile full --seed 42
    & $PythonStr simulations/src/plot_nested_hs_suite.py
}
elseif ($Command -eq "sim-qubit") {
    & $PythonStr simulations/src/run_nested_hs_suite.py --regime cng --profile full --seed 42
    & $PythonStr simulations/src/plot_nested_hs_suite.py
}
elseif ($Command -eq "paper") {
    Push-Location manuscript/tex
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex
    Pop-Location
}
elseif ($Command -eq "clean") {
    Remove-Item -Recurse -Force manuscript/build/*
}
else {
    Write-Host "Usage: ./manage.ps1 [install|sim|sim-qutrit|sim-qubit|paper|clean]"
}
