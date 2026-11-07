"""Verify the byte-frozen pre-Phase-19 prefix and production trees."""
from src.evaluation import phase12_prospective as p
from src.evaluation import phase19_authorization as auth

BEFORE = auth.RUN / 'protected_before.json'
PRODUCTION = auth.RUN / 'production_before.json'


def verify():
    protected = p.read(BEFORE)
    missing = [path for path in protected if not (p.ROOT / path).is_file()]
    changed = [path for path, digest in protected.items()
               if (p.ROOT / path).is_file() and p.source.sha256_file(p.ROOT / path) != digest]
    production = p.read(PRODUCTION)
    production_changes = {}
    for base, files in production.items():
        root = p.ROOT / base
        current = {path.relative_to(p.ROOT).as_posix(): p.source.sha256_file(path)
                   for path in root.rglob('*') if path.is_file() and
                   path.suffix not in ('.pyc', '.lock', '.tmp')}
        before = {key: value for key, value in files.items() if
                  not key.endswith(('.pyc', '.lock', '.tmp'))}
        unexpected = sorted(set(current) - set(before))
        deleted = sorted(set(before) - set(current))
        drift = sorted(key for key in set(before) & set(current) if before[key] != current[key])
        if unexpected or deleted or drift:
            production_changes[base] = {'added': unexpected, 'deleted': deleted, 'changed': drift}
    return p.seal({'schema_version': 'ffps.phase19_protected_integrity.v1',
        'protected_files': len(protected), 'missing': missing, 'changed': changed,
        'production_changes': production_changes,
        'verified': not (missing or changed or production_changes)}, 'record_sha256')
