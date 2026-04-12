@echo off
cd /d C:\Users\papol\my_systems\betting_partner_system
set PYTHONPATH=C:\Users\papol\my_systems\betting_partner_system
python scripts\nightly_pipeline.py >> C:\Users\papol\my_systems\betting_partner_system\logs\nightly.log 2>&1
