# Approved matcher test pin supersession

The owner approved this supersession on 2026-11-25 (SAST). It applies to `tests/data_foundation/test_approved_matcher.py` in the reviewed release stack.

| Recorded file version | SHA-256 |
| --- | --- |
| Historical NEXT-38 packaging pin from `1592957` | `acad108386b46f17dbe23e758b8492e2adae458f3608d4c7715e41ef0bceebfc` |
| Inherited test at `13f3d7e` | `84180552f638cc6a8dbd108dcc989bc157d55e6ea6171f19eba025756f8976c7` |
| Approved replacement test | `4a68f34249ea2d676e6a6fd7d06818ddb015871f94336422f67e7f2a9eb05cfe` |

The replacement asserts the two literal quarantine keys with two occurrences each and the four exact source locations, lines 179, 182, 295, and 296. Expectations derived from the loaded decisions could miss a fixture or loader change that moved rows or changed the key distribution while keeping the total at four.

The historical NEXT-38 packaging pin and the retained records still apply to their original evidence. This successor governs the release test only.
