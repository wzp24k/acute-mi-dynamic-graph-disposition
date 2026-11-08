$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$chineseOutputs = Get-ChildItem -LiteralPath (Join-Path $root "outputs") -File |
    Where-Object {
        $_.Name -like "CTOGT_MACCE_*" -and
        $_.Name -notmatch "manuscript|supplement" -and
        $_.Extension -in @(".docx", ".md")
    } |
    Sort-Object Name
$targetPaths = @(
    "outputs\CTOGT_MACCE_manuscript.docx",
    "outputs\CTOGT_MACCE_manuscript.md",
    "outputs\CTOGT_MACCE_supplement.docx",
    "outputs\POSITIVE_RESULT_SUMMARY.md",
    "outputs\TRAINING_STATUS.md",
    "outputs\CTOGT_MACCE_submission_bundle.zip",
    "results\final_qc.json",
    "results\temporal_holdout_validation.json",
    "results\nested_selection_and_utility.json",
    "results\nested_transformer_selection.json",
    "results\strong_temporal_holdout_fusion.json",
    "results\strong_external_transport.json",
    "results\clinical_utility_analysis.json",
    "results\interval_threshold_sensitivity.json",
    "results\eicu_site_holdout.json",
    "results\metrics_external_eicu_anycad_temporal_ensemble.json",
    "results\manual_endpoint_review_summary.json"
) | ForEach-Object { Join-Path $root $_ }
$targetPaths += $chineseOutputs.FullName

$lines = foreach ($path in $targetPaths) {
    $relative = $path.Substring($root.Length + 1)
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    "$hash  $relative"
}

[System.IO.File]::WriteAllLines(
    (Join-Path $root "outputs\SHA256SUMS.txt"),
    $lines,
    (New-Object System.Text.UTF8Encoding($false))
)
Get-Content -LiteralPath (Join-Path $root "outputs\SHA256SUMS.txt")
