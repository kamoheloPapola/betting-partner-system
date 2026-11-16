"""Byte-preserving candidate transforms. Outcome cells are opaque bytes."""
import csv
import hashlib
import io


class RepairError(ValueError):
    pass


def fields(raw):
    """Split one physical CSV record, retaining each original field token."""
    raw = raw.rstrip(b"\r\n")
    output, start, quoted, i = [], 0, False, 0
    while i < len(raw):
        if raw[i] == 34:
            if quoted and i+1 < len(raw) and raw[i+1] == 34:
                i += 2
                continue
            quoted = not quoted
        elif raw[i] == 44 and not quoted:
            output.append(raw[start:i])
            start = i+1
        i += 1
    if quoted:
        raise RepairError("Multiline/malformed CSV record refused")
    output.append(raw[start:])
    return output


def encoded_cell(text):
    out = io.StringIO()
    csv.writer(out, lineterminator="").writerow([str(text)])
    return out.getvalue().encode("utf-8")


def deduplicate_identical(raw):
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise RepairError("Empty candidate source")
    seen, output, trace = {}, [lines[0]], []
    for line, value in enumerate(lines[1:], 2):
        # No field decoding or outcome comparison: only exact byte identity.
        if value in seen:
            trace.append({"source_line": line, "kept_source_line": seen[value],
                          "row_sha256": hashlib.sha256(value).hexdigest()})
        else:
            seen[value] = line
            output.append(value)
    return b"".join(output), {"input_rows": len(lines)-1, "output_rows": len(output)-1,
                             "removed_identical_rows": len(trace), "duplicates": trace,
                             "outcome_cells_decoded": False}


def relabel_2025(raw, plans, provider_by_id):
    lines = raw.splitlines(keepends=True)
    if not lines:
        raise RepairError("Empty candidate source")
    header = next(csv.reader([lines[0].decode("utf-8-sig")]))
    for key in ("home_team", "away_team"):
        if header.count(key) != 1:
            raise RepairError("Identity column missing/duplicated")
    appended = ("home_provider_id", "away_provider_id", "provider_match_id")
    if any(k in header for k in appended):
        raise RepairError("Refuse already-repaired source")
    by_line = {r["line"]: r for r in plans}
    if len(by_line) != len(plans) or len(plans) != len(lines)-1:
        raise RepairError("Plans must cover every source row exactly once")
    source_sha = hashlib.sha256(raw).hexdigest()
    if any(p.get("source_sha256") != source_sha for p in plans):
        raise RepairError("Plan/source identity mismatch")
    newline = b"\r\n" if lines[0].endswith(b"\r\n") else b"\n"
    output = [lines[0].rstrip(b"\r\n")+b","+b",".join(k.encode() for k in appended)+newline]
    quarantine = [lines[0]]
    trace = []
    hi, ai = header.index("home_team"), header.index("away_team")
    for line, original in enumerate(lines[1:], 2):
        plan = by_line[line]
        if plan["status"] == "EXCLUDED":
            quarantine.append(original)
            trace.append({"source_line": line, "disposition": "EXCLUDED",
                          "original_row_sha256": hashlib.sha256(original).hexdigest()})
            continue
        if plan["status"] != "MATCHED":
            raise RepairError("Unresolved row cannot enter a repaired candidate")
        provider = provider_by_id[plan["fixture_id"]]
        if (provider["home_id"],provider["away_id"]) != (plan["home_id"],plan["away_id"]):
            raise RepairError("Provider identity/plan disagreement")
        old = fields(original)
        if len(old) != len(header):
            raise RepairError("Unexpected field count; no repair written")
        new = list(old)
        new[hi] = encoded_cell(provider["home_meta"]["name"])
        new[ai] = encoded_cell(provider["away_meta"]["name"])
        if any(old[i] != new[i] for i in range(len(old)) if i not in (hi, ai)):
            raise RepairError("Nonidentity cell changed")
        extra = [str(plan["home_id"]).encode(),str(plan["away_id"]).encode(),str(plan["fixture_id"]).encode()]
        output.append(b",".join(new+extra)+newline)
        trace.append({"source_line": line, "candidate_line": len(output),
                      "provider_fixture_id": plan["fixture_id"], "nonidentity_tokens_equal": True,
                      "original_row_sha256": hashlib.sha256(original).hexdigest()})
    return b"".join(output), b"".join(quarantine), {
        "input_rows": len(lines)-1, "output_rows": len(output)-1, "excluded_rows": len(quarantine)-1,
        "nonidentity_tokens_equal": True, "outcome_cells_decoded": False, "rows": trace}

