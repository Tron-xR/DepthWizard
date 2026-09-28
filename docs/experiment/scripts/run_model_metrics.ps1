$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$py = "C:\Users\harsh\Downloads\files\DepthWizard\server\.venv\Scripts\python.exe"
$exp = "C:\Users\harsh\AppData\Local\Temp\opencode\exp"
$log = "C:\Users\harsh\AppData\Local\Temp\opencode\exp\run_model_metrics.log"
Remove-Item $log -ErrorAction SilentlyContinue
foreach ($spec in @(
    @{a="B250"; s=0; t="native"}, @{a="B250"; s=1; t="native"}, @{a="B250"; s=2; t="native"},
    @{a="D90";  s=0; t="native"}, @{a="D90";  s=1; t="native"}, @{a="D90";  s=2; t="native"},
    @{a="D250"; s=0; t="native"}, @{a="D250"; s=1; t="native"}, @{a="D250"; s=2; t="native"},
    @{a="B90";  s=0; t="dem3dep"}, @{a="B90";  s=1; t="dem3dep"}, @{a="B90";  s=2; t="dem3dep"},
    @{a="B250"; s=0; t="dem3dep"}, @{a="B250"; s=1; t="dem3dep"}, @{a="B250"; s=2; t="dem3dep"},
    @{a="D90";  s=0; t="dem3dep"}, @{a="D90";  s=1; t="dem3dep"}, @{a="D90";  s=2; t="dem3dep"},
    @{a="D250"; s=0; t="dem3dep"}, @{a="D250"; s=1; t="dem3dep"}, @{a="D250"; s=2; t="dem3dep"}
)) {
    $pre = if ($spec.t -eq "dem3dep") { "metrics3dep_" } else { "metrics_" }
    $out = Join-Path $exp ("runs\{0}{1}_{2}.csv" -f $pre, $spec.a, $spec.s)
    if (Test-Path $out) {
        Add-Content $log "skip $($spec.a) s$($spec.s) $($spec.t) (exists)"
        continue
    }
    Add-Content $log "START $($spec.a) s$($spec.s) $($spec.t) $(Get-Date -Format 'HH:mm:ss')"
    & $py "$exp\exp_run_metrics.py" --arm $spec.a --seed $spec.s --truth $spec.t 2>&1 | ForEach-Object { $_.Replace([string][char]0, "") } | Out-File -Append -Encoding utf8 $log
    Add-Content $log "END $($spec.a) s$($spec.s) $($spec.t) rc=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')"
}
Add-Content $log "ALL MODEL METRICS DONE $(Get-Date -Format 'HH:mm:ss')"