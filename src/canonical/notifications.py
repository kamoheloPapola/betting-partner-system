"""Best-effort local desktop failure notification; never contains credentials."""
import os
import re
import subprocess
import sys
import time


def notify_local(code):
    code=code if isinstance(code,str) and re.fullmatch(r"[A-Z0-9_]{1,100}",code) else "CANONICAL_JOB_FAILED"
    if sys.platform!="win32":
        return {"attempted":False,"delivered":False,"reason":"WINDOWS_DESKTOP_UNAVAILABLE"}
    # Pump shell messages so a successful request can be distinguished from a displayed balloon.
    script=(
        "$ErrorActionPreference='Stop';"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$icon=New-Object System.Windows.Forms.NotifyIcon;"
        "$script:bpsShown=$false;"
        "try{"
        "$icon.Icon=[System.Drawing.SystemIcons]::Warning;"
        "$icon.add_BalloonTipShown({$script:bpsShown=$true});"
        "$icon.Visible=$true;"
        "$icon.BalloonTipTitle='Betting Partner needs attention';"
        "$icon.BalloonTipText='"+code+" - open the local dashboard';"
        "$icon.ShowBalloonTip(5000);"
        "$timer=[System.Diagnostics.Stopwatch]::StartNew();"
        "while($timer.ElapsedMilliseconds -lt 7000){"
        "[System.Windows.Forms.Application]::DoEvents();Start-Sleep -Milliseconds 100};"
        "if($script:bpsShown){Write-Output 'BALLOON_SHOWN'}"
        "}finally{$icon.Dispose()}"
    )
    command=[os.path.join(os.environ.get("SYSTEMROOT",r"C:\Windows"),"System32/WindowsPowerShell/v1.0/powershell.exe"),
             "-NoLogo","-NoProfile","-NonInteractive","-Command",script]
    started=time.monotonic()
    try:
        cp=subprocess.run(command,capture_output=True,timeout=10,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        return {"attempted":True,"delivery_requested":cp.returncode==0,"exit_code":cp.returncode,
                "duration_seconds":round(time.monotonic()-started,3),
                "display_confirmed":cp.returncode==0 and b"BALLOON_SHOWN" in cp.stdout.splitlines()}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {"attempted":True,"delivery_requested":False,"display_confirmed":False,
                "error_type":type(exc).__name__}

