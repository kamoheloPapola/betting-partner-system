# Dedicated action using this repository's existing Windows Task Scheduler convention.
# Run after tests, a normal operation, and offline verification have passed.
$ErrorActionPreference = 'Stop'
$phase12Root = Split-Path -Parent $PSScriptRoot
$phase12Python = Join-Path $phase12Root 'venv\Scripts\pythonw.exe'
$phase12Script = Join-Path $phase12Root 'scripts\run_phase12_operations.py'
$phase12Name = 'FFPS Phase 12 Prospective Collection'
if (-not (Test-Path -LiteralPath $phase12Python)) { throw 'Repository Python runtime missing' }
$phase12Action = New-ScheduledTaskAction -Execute $phase12Python -Argument ('"' + $phase12Script + '" run') -WorkingDirectory $phase12Root
$phase12Existing = Get-ScheduledTask -TaskName $phase12Name -ErrorAction SilentlyContinue
if ($phase12Existing) {
    if ($phase12Existing.Actions.Execute -ne $phase12Action.Execute -or $phase12Existing.Actions.Arguments -ne $phase12Action.Arguments) {
        throw 'Existing task differs; refusing to replace another action'
    }
    $phase12Existing
    return
}
$phase12Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddHours(1) -RepetitionInterval (New-TimeSpan -Hours 1)
$phase12Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 6) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$phase12User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$phase12Principal = New-ScheduledTaskPrincipal -UserId $phase12User -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $phase12Name -Action $phase12Action -Trigger $phase12Trigger -Settings $phase12Settings -Principal $phase12Principal -Description 'Frozen SA FL1 PD BL1 prospective collection and settlement only. No training or activation.'
