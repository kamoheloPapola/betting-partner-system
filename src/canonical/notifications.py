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
    # Fixed message and a bounded child; no machine/user notification policy is changed.
    script=(
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$icon=New-Object System.Windows.Forms.NotifyIcon;"
        "$icon.Icon=[System.Drawing.SystemIcons]::Warning;"
        "$icon.Visible=$true;"
        "$icon.BalloonTipTitle='Betting Partner needs attention';"
        "$icon.BalloonTipText='"+code+" - open the local dashboard';"
        "$icon.ShowBalloonTip(5000);Start-Sleep -Seconds 6;$icon.Dispose()"
    )
    command=[os.path.join(os.environ.get("SYSTEMROOT",r"C:\Windows"),"System32/WindowsPowerShell/v1.0/powershell.exe"),
             "-NoLogo","-NoProfile","-NonInteractive","-Command",script]
    started=time.monotonic()
    try:
        cp=subprocess.run(command,capture_output=True,timeout=10,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        return {"attempted":True,"delivery_requested":cp.returncode==0,"exit_code":cp.returncode,
                "duration_seconds":round(time.monotonic()-started,3),"confirmation":"Desktop visibility is not attested"}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {"attempted":True,"delivery_requested":False,"error_type":type(exc).__name__}

