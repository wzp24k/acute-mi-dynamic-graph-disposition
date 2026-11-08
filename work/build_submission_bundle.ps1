$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$zipPath = Join-Path $root "outputs\CTOGT_MACCE_submission_bundle.zip"
$tempPath = Join-Path $root "work\CTOGT_MACCE_submission_bundle.tmp.zip"

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression
if (Test-Path -LiteralPath $tempPath) {
    [System.IO.File]::Delete($tempPath)
}
$zip = [System.IO.Compression.ZipFile]::Open(
    $tempPath,
    [System.IO.Compression.ZipArchiveMode]::Create
)

$paths = @(
    (Join-Path $root "README.md"),
    (Join-Path $root "configs"),
    (Join-Path $root "docs"),
    (Join-Path $root "outputs"),
    (Join-Path $root "results"),
    (Join-Path $root "src"),
    (Join-Path $root "tests"),
    (Join-Path $root "work"),
    (Join-Path $root "data\processed")
)

$files = foreach ($path in $paths) {
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path
        if ($item.PSIsContainer) {
            Get-ChildItem -LiteralPath $path -Recurse -File
        }
        else {
            $item
        }
    }
}

$exclude = @(
    "\__pycache__\",
    "\.pytest_cache\",
    "CTOGT_MACCE_submission_bundle.zip",
    "CTOGT_MACCE_submission_bundle.tmp.zip",
    "SHA256SUMS.txt"
)

foreach ($file in $files) {
    $excluded = $false
    foreach ($pattern in $exclude) {
        if ($file.FullName.Contains($pattern)) {
            $excluded = $true
            break
        }
    }
    if ($excluded) {
        continue
    }
    $relative = $file.FullName.Substring($root.Length + 1).Replace("\", "/")
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
        $zip,
        $file.FullName,
        $relative,
        [System.IO.Compression.CompressionLevel]::Optimal
    ) | Out-Null
}

$zip.Dispose()
[System.IO.File]::Copy($tempPath, $zipPath, $true)
[System.IO.File]::Delete($tempPath)
Get-Item -LiteralPath $zipPath |
    Select-Object FullName, Length, LastWriteTime |
    Format-List
