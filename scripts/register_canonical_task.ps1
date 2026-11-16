# Register only after an adopted complete model and successful live smoke.
# No credentials in task arguments; DPAPI decrypts only as this interactive user.
[CmdletBinding()]
param([string]$DailyAt = '18:00')
$ErrorActionPreference = 'Stop'
$taskName = 'BettingPartnerCanonical'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $repoRoot 'venv\Scripts\python.exe'
$runnerPath = Join-Path $repoRoot 'scripts\run_canonical.py'
if ($DailyAt -notmatch '^\d{2}:\d{2}$') { throw 'DailyAt must be HH:mm local time.' }
$triggerTime = [datetime]::ParseExact($DailyAt,'HH:mm',[Globalization.CultureInfo]::InvariantCulture)
# A bounded child handles all local readiness checks; no provider request.
$startInfo = New-Object Diagnostics.ProcessStartInfo
$startInfo.FileName = $pythonPath
$startInfo.Arguments = '-B "' + $runnerPath + '" scheduler-ready'
$startInfo.WorkingDirectory = $repoRoot
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$process = [Diagnostics.Process]::Start($startInfo)
if (-not $process.WaitForExit(15000)) {
    $process.Kill()
    throw 'Readiness check timed out; no task registered.'
}
if ($process.ExitCode -ne 0) { throw 'Readiness gates failed; no task registered.' }
$before = @(Get-ScheduledTask | Select-Object TaskName,TaskPath,State)
$legacy = @($before | Where-Object { $_.TaskName -eq 'BettingPartnerNightly' })
if ($legacy.Count -ne 1 -or $legacy[0].State -ne 'Disabled') {
    throw 'Expected exactly one disabled legacy task; no task registered.'
}
if (@($before | Where-Object { $_.TaskName -eq $taskName }).Count -ne 0) {
    throw 'Canonical task already exists; refusing replacement.'
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument ('-B "' + $runnerPath + '" run --stored-key') -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $triggerTime
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -TaskPath '\' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Canonical local goals-only fetch/freeze/settle. No training or cleanup.' | Out-Null
$after = @(Get-ScheduledTask | Select-Object TaskName,TaskPath,State)
$oldAfter = @($after | Where-Object { $_.TaskName -ne $taskName })
$delta = Compare-Object ($before | Sort-Object TaskPath,TaskName | ConvertTo-Json -Compress) ($oldAfter | Sort-Object TaskPath,TaskName | ConvertTo-Json -Compress)
if ($delta) { throw 'An existing task changed during registration; inspect task state. No automatic rollback attempted.' }
[pscustomobject]@{TaskName=$taskName;State=(@($after | Where-Object {$_.TaskName -eq $taskName})[0].State);User=$identity;DailyAtLocal=$DailyAt;LegacyTask='Disabled'} | ConvertTo-Json -Compress

