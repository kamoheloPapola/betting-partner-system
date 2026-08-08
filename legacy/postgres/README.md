# Legacy PostgreSQL path

PostgreSQL was disabled during state-model consolidation because the documented
deployment never connected the API or scheduler to it, while manifest, resolved
prediction, and drift writes were already authoritative in files.

`migrate_to_postgres.py` is retained only for audit history. It was a one-way
import into PostgreSQL, not an export or a supported migration path.

Before upgrading any external deployment that independently supplied
`DATABASE_URL`, an operator must inspect and export that database. The repository
cannot establish whether an externally managed Render/dashboard deployment has
such data. The documented Docker Compose topology had no application connection
to PostgreSQL and no repository-owned PostgreSQL data was found.
