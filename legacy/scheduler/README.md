# Legacy scheduler entrypoints

These files are retained only to document the scheduler paths that were
superseded during deployment-contract consolidation. They are unused and must not
be installed or executed.

The canonical pipeline is `python -m src.scheduler.nightly` in
`Dockerfile.scheduler`. The only recurring trigger is
`deploy/cron/betting-nightly`, which invokes `scripts/run_scheduler_container.sh`.

The retired files diverged in commands, timing, error handling, runtime, or paths,
so leaving them in active locations risked accidental execution of a different
pipeline.
