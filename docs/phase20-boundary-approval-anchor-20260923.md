# Phase 20 boundary approval anchor

The user approval brief (file SHA-256
`477cc983c709fcfe10f27b9f6e84993da31e121cc0e4dc7656a2ad99cfc327d4`)
directed issuance of a separate APPROVED contract for the exact reviewed v7
candidate. The issuer checked v7 SHA-256
`b8dddc14f72f8b26aa3a1a9fa9e54f86b8ee3f23705f90ac94f25aba559835a4`
and replay certificate SHA-256
`a16e8492fa243e344625a7396abdbc916b0c03c15789ae7fd5c35d2575c9b4f9`.
It published [the new contract](phase20-boundary-approved-v1.json) with
write-once creation. V7 was not overwritten.

The approved contract's **whole-file SHA-256** was measured separately from
the issuer and verifier with PowerShell `Get-FileHash -Algorithm SHA256`:

```text
1ed13b93c37bd18b17a52a256c4d8358abdb0269c8574e7e9f2fe44994b63876
```

This hash was supplied as the explicit `--approved-sha256` argument to the
independent verifier. It is not an internal contract seal and is not inferred
by the verifier. The user-facing approval record also carries this exact hash
outside the approved contract.

The verifier returned `verified:true`, `protected_existing:19176`,
`changed:[]`, `missing:[]`, 167 authorized append-only additions, 12
quarantined historical files, 4 terminated-incomplete files, and zero
unauthorized, unknown, or unfinished additions. The same real contract
returned `verified:false` with no hash and with a wrong hash, both with
`BOUNDARY_APPROVAL_HASH_REQUIRED`.

Isolated fixture tests also reject a modified approved contract, changed or
missing frozen file, changed authorized addition, and unexpected addition.
The focused boundary, Phase 20 settlement, and Local Operator suite passed
42 tests.

Approval covers physical coexistence of these exact 183 additions. It does
not clear FL1 event 102, clean report 33, complete report 34, include PD
Phase 12 event 118 in Phase 19 settlement, authorize new writes, or promote a
model. The live Phase 20 integrity gate and settlement path are unchanged.
