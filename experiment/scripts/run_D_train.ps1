$ErrorActionPreference = "Continue"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$py = "C:\Users\harsh\Downloads\files\DepthWizard\server\.venv\Scripts\python.exe"
$exp = "C:\Users\harsh\AppData\Local\Temp\opencode\exp"
$log = "C:\Users\harsh\AppData\Local\Temp\opencode\exp\run_D.log"
Remove-Item $log -ErrorAction SilentlyContinue
foreach ($arm in @("D90", "D250")) {
    foreach ($seed in @(0, 1, 2)) {
        $ckpt = Join-Path $exp ("runs\{0}" -f $arm)
        $cvp = Join-Path $exp ("runs\curves_{0}_{1}.csv" -f $arm, $seed)
        $done = (Test-Path (Join-Path $ckpt "s${seed}_f3_best.pt")) -and
                (Test-Path $cvp) -and ((Get-Content $cvp | Measure-Object -Line).Lines -ge 119)
        if ($done) { Add-Content $log "[$arm s$seed] already done - skip"; continue }
        Add-Content $log "=== START $arm s$seed $(Get-Date -Format 'HH:mm:ss') ==="
        & $py "$exp\exp_train.py" --arm $arm --seed $seed --noeval *>> $log
        Add-Content $log "=== END $arm s$seed rc=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss') ==="
    }
}
Add-Content $log "ALL D TRAINING DONE $(Get-Date -Format 'HH:mm:ss')"