# LEGACY/UNUSED daily agent workflow

Superseded by `Dockerfile.scheduler` and `deploy/cron/betting-nightly`. Retained
for audit history only; these commands are not the supported production pipeline.

## Former morning routine

```bash
python -m src.cli.main fetch-latest-season
python -m src.cli.main ingest-results
python -m src.cli.main resolve-predictions
python -m src.cli.main show-predictions
python -m src.cli.main forbidden-fruit
```

## Former weekly maintenance

```bash
python -m src.cli.main check-drift
python -m src.cli.main refresh-drift --window 30
python -m src.cli.main audit-coverage
```

## Former pre-season maintenance

The retired workflow instructed an operator to unlock the system, refresh team
offsets, train Poisson and negative-binomial models, run league backtests, and
lock the system again. It was advisory agent documentation, not an executable or
reviewable deployment schedule.
