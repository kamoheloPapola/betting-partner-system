# Phase 12 historical writer freeze and stable boundary capture

Decision date: 2026-09-23. This freezes the **automatic historical Phase 12
writer** for the personal-PC workflow. It does not delete or change Phase 12
evidence, clear any protocol violation, complete report 34, or change the
Phase 20 settlement gate.

## Writer and invocation

The only scheduled task whose action invokes the Phase 12 runners was
`\FFPS Phase 12 Prospective Collection`. Its action was exactly this
repository's `venv\Scripts\pythonw.exe` executing
`scripts\run_phase12_operations.py run` with the repository as working
directory. The task was `Ready` before intervention and is now **Disabled**.
An elevated process check found no running `run_phase12_operations.py`
invocation after disablement. This stops automatic future launches; a person
could still explicitly invoke the historical runner, so the boundary must be
recaptured if that happens.

Task Scheduler's local history identifies instance
`{6ec2d1e4-76cf-4241-897f-7fe9f860131e}`. It launched at
2026-09-23 13:50:03 South Africa time (11:50:03 UTC) with the exact action
above. Scheduler event 329 says it was **terminated for exceeding the
configured six-hour execution limit** at 19:50:08 local (17:50:08 UTC).
Overlapping launches at 18:05, 19:13, and 19:50 local were ignored because
that instance was already running. The task was disabled at 20:24:44 local
(18:24:44 UTC), recorded by Scheduler event 142. These records identify the
scheduled invocation and its timeout, though they are not per-file OS write
audit records.

Report 34's intent has key `20260923T11` and was created at 11:50:35 UTC,
within that task launch. Its SA claim was sealed at 17:20:11 UTC. The two
additions first seen after the 181-file inventory are:

| File | Recorded effect |
|---|---|
| `SA/checkpoints/97da214474eb9f08045f7dcad7012dcbcf744d37b5d9e64312ad11e51a571dc5.json` | Checkpoint as of 17:37:45 UTC at unchanged SA journal head 121; 34 pairs, 19 settled, 15 pending; `replay: VERIFIED`, `scientific_state: PROTOCOL_VIOLATION_BLOCKED`. |
| `operational/report-000034/SA/outcome.json` | Sealed after that checkpoint; zero newly admitted pairs, zero newly settled fixtures, no new incidents, and no error. |

Those paths are relative to the frozen Phase 12 run directory. The file
timestamps, report 34 claim and checkpoint references, scheduler action,
instance timing, and `phase12_operations.process_league` publication path
jointly attribute the additions to report 34's scheduled operation. The
available logs do not give a separate file-write process ID. Report 34 has
**no final `report.json`**. Its invocation timed out; the package remains
unfinished and is not represented as a completed operational report. The SA
historical protocol violation remains blocked.

## Stable capture after disablement

The fresh per-file inventory is
`docs/phase20-protected-boundary-inventory-after-writer-freeze.json`.
The first scan ran 18:25:39–18:26:10 UTC; a second independent scan ran
18:32:12–18:32:38 UTC. Both found **183 additions**, **zero changed** and
**zero missing** among the original 19,176 protected files. Neither scan saw
a file arrive or disappear during its window. A separate comparison checked
that all 183 added paths and SHA-256 values matched between captures.
This establishes an observed stable interval, not a permanent filesystem
lock.

The new write-once proposal
`docs/phase20-boundary-authorization-proposal-v5.json` (file SHA-256
`9c26e8ba16cd88c4895dea1b8c6bc9718e103b6296182226f94bacb8b69dfc46`)
binds that exact 183-file inventory. It remains **PENDING** and grants **zero**
append-only authorizations: 178 additions are unauthorized candidates, two
FL1 files are quarantined, and the three report 34 files are unfinished.
The independent verifier returned `verified: false` with zero unknown
additions and `BOUNDARY_APPROVAL_REQUIRED`. The live Phase 20 integrity gate
still rejects the additions and settlement remains blocked.

No Phase 12 pair, checkpoint, source archive, journal entry, or report file
was edited or removed by this freeze. The Phase 19 pair heads and all 1,172
recorded predictions remain outside the historical writer's state changes.
