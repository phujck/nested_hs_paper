# flatten_arxiv.ps1
# Automates the creation of a flattened arXiv submission package.

$ErrorActionPreference = "Stop"

$sourceDir = "manuscript/tex"
$figuresDir = "simulations/results/figures"
$buildDir = "manuscript/build"
$outputDir = "arxiv_submission_flat"
$zipFile = "arxiv_submission_flat.zip"

Write-Host "Cleaning up old submission files..."
if (Test-Path $outputDir) { Remove-Item -Recurse -Force $outputDir }
if (Test-Path $zipFile) { Remove-Item -Force $zipFile }

New-Item -ItemType Directory -Force $outputDir | Out-Null

Write-Host "Copying source files..."
# Copy main.tex
Copy-Item "$sourceDir/main_v2.tex" "$outputDir/main.tex"

# Copy sections (flattened)
Get-ChildItem "$sourceDir/sections/*.tex" | ForEach-Object {
    Copy-Item $_.FullName "$outputDir/$($_.Name)"
}

# Copy bbl (if exists)
if (Test-Path "$buildDir/main_v2.bbl") {
    Copy-Item "$buildDir/main_v2.bbl" "$outputDir/main.bbl"
} else {
    Write-Warning "No .bbl file found in $buildDir. You may need to compile the manuscript first."
}

# Copy figures (flattened)
Get-ChildItem "$figuresDir/*.pdf" | ForEach-Object {
    Copy-Item $_.FullName "$outputDir/$($_.Name)"
}

Write-Host "Rewriting file paths in TeX files..."

# Function to rewrite paths in a file
function Rewrite-In-File {
    param (
        [string]$FilePath
    )
    
    $content = Get-Content $FilePath -Raw
    
    # 1. Flatten \input{sections/...} to \input{...}
    # Matches \input{sections/filename} or \input{sections/filename.tex}
    $content = $content -replace '\\input\{sections/([^}]+)\}', '\input{$1}'
    
    # 2. Flatten \includegraphics{.../figures/...} to \includegraphics{...}
    # Matches \includegraphics[...]{../../.../filename.pdf} or similar
    # This regex looks for the last part of the path in the second brace argument
    $content = [Regex]::Replace($content, '\\includegraphics(\[.*?\])?\{.*[/\\](.+?)\}', {
        param($match)
        $opts = $match.Groups[1].Value
        $filename = $match.Groups[2].Value
        return "\includegraphics$opts{$filename}"
    })

    Set-Content -Path $FilePath -Value $content -NoNewline
}

# Process main.tex
Rewrite-In-File "$outputDir/main.tex"

# Process all section files
Get-ChildItem "$outputDir/*.tex" | Where-Object { $_.Name -ne "main.tex" } | ForEach-Object {
    Rewrite-In-File $_.FullName
}

Write-Host "Removing unused/temporary files..."
# Remove any tex files that are not strictly needed if we want to be clean, 
# but copying all sections is usually safer. 
# We can remove specifically identified unused ones if needed.

Write-Host "Zipping submission package..."
Compress-Archive -Path "$outputDir\*" -DestinationPath $zipFile -Force

Write-Host "Done! Submission package created at $zipFile"
