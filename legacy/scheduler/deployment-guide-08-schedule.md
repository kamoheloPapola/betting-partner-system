# LEGACY/UNUSED direct prediction cron

Superseded by `Dockerfile.scheduler` and `deploy/cron/betting-nightly`. Retained
for audit history only. Do not install this cron entry because it runs only the
prediction command rather than the canonical seven-step nightly pipeline.

Former deployment-guide instruction:

```cron
0 8 * * * cd /path/to/app && python -m src.cli show-predictions --date today >> logs/daily.log 2>&1
```
