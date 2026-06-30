param(
  [string]$Config = "configs/wmdp_safe_port_template.json"
)

$ErrorActionPreference = "Stop"

Write-Host "SAFE-PoRT example commands. Review hardware requirements before running."
Write-Host "Config: $Config"
Write-Host ""
Write-Host "safe-port build-data --config $Config"
Write-Host "safe-port mine-beliefs --config $Config"
Write-Host "safe-port train-adapter --config $Config"
Write-Host "safe-port evaluate --config $Config"

