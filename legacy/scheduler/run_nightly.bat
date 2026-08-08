@echo off
REM LEGACY/UNUSED: superseded by Dockerfile.scheduler and deploy/cron/betting-nightly.
REM Retained for audit history only. Do not execute.
cd /d C:\Users\papol\my_systems\betting_partner_system
set PYTHONPATH=C:\Users\papol\my_systems\betting_partner_system
python scripts\nightly_pipeline.py >> C:\Users\papol\my_systems\betting_partner_system\logs\nightly.log 2>&1
