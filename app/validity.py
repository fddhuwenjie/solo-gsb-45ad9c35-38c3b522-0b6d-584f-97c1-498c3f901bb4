"""Continuous validity review of a registered, immutable admission proof.

A registered admission attestation can stop satisfying its original
admission conditions as time advances or materials are supplemented
(e.g. a previously open deadline becomes overdue).  A validity review
*never rewrites the stored proof*: the attestation and its frozen release
batch are loaded together in one SQLite **read-only** transaction, the
bindings and rule structure are proven unchanged, and the fulfillment
snapshot is recomputed from the submitted ``now`` and materials.

The caller submits the binding the proof was minted with
(``batch_id``, ``batch_hash``) and the snapshot inputs: the new ``now``,
the original ``release_at``, timezone, workdays/holidays, per-obligation
deadline rules and the new material/evidence set.  Only **time
(``now``) and the materials** may differ from the minted proof; the
frozen batch hash, ``release_at``, timezone, workdays, holidays, the
deadline-rule structure and the nested evidence rules must stay
identical, otherwise the review is refused as a binding conflict
(409).  An unknown attestation is 404; malformed envelope input is 400;
invalid scheduling/evidence content is 422.

The conclusion is one of the four stable strings:

* ``unchanged``    -- every instance is still ``satisfied`` and the
  recomputed snapshot is byte-identical to the one embedded in the
  proof (i.e. the review is a pure replay: same normalized ``now`` and
  materials);
* ``valid-changed`` -- every instance is still ``satisfied`` but the
  snapshot changed (allowed inputs moved: time and/or materials);
* ``pending``      -- at least one instance is not yet ``satisfied``
  while none is overdue (a deadline is still open, e.g. evidence
  pending);
* ``expired``      -- at least one instance is ``overdue`` -- the
  admission no longer holds.

``unchanged``/``valid-changed`` take precedence over ``pending``/
``expired`` only in the all-satisfied case; any overdue instance forces
``expired`` regardless of satisfied witnesses.

The response carries the per-instance differences versus the embedded
snapshot (``diff_instances``, only instances whose status or witness
set moved) and a ``review_hash`` over the canonical conclusion.  Every
collection is sorted, so object/array key order in the request never
changes the conclusion, the diff or the hash.
"""

import re

from .common import ClientError, ConflictError, stable_hash
from .snapshot import compute_batch_snapshot

_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# Snapshot fields whose presence is asserted by the envelope; their
# content validation is delegated to the snapshot builder (422).
_REQUIRED_FIELDS = (
    "batch_id",
    "batch_hash",
    "now",
    "release_at",
    "timezone",
    "workdays",
    "holidays",
    "obligations",
)


def validate_review_body(body):
    """Validate the validity-review envelope (HTTP 400 rules only).

    Returns ``(batch_id, expected_batch_hash, body)``; every
    scheduling/evidence/material content failure is surfaced later by
    the snapshot builder as HTTP 422.
    """
    if not isinstance(body, dict):
        raise ClientError("request body must be a JSON object")

    batch_id = body.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ClientError("batch_id must be a non-empty string")
    batch_id = batch_id.strip()
    if not _BATCH_ID_RE.match(batch_id):
        raise ClientError(
            "batch_id may contain only letters, digits, dot, underscore"
            " and hyphen (1..128 characters)"
        )

    expected_hash = body.get("batch_hash")
    if not isinstance(expected_hash, str) or not _HEX64_RE.match(
        expected_hash.strip()
    ):
        raise ClientError(
            "batch_hash must be the 64 lowercase hex characters of the"
            " batch hash bound to the attestation"
        )
    expected_hash = expected_hash.strip()

    for field in _REQUIRED_FIELDS:
        if field not in body:
            raise ClientError("validity review requires %r" % field)

    return batch_id, expected_hash, body


def _original_materials(original_snapshot):
    """Reconstruct snapshot-builder-shaped materials from the proof's
    embedded snapshot (its normalized output round-trips as snapshot
    input: the extra ``provided_date`` key is ignored on re-entry)."""
    return [
        {
            "id": m["id"],
            "digest": m["digest"],
            "type": m["type"],
            "scope": m["scope"],
            "component": m["component"],
            "provided_at": m["provided_at"],
        }
        for m in original_snapshot.get("materials", [])
    ]


def _original_fixed_inputs(original_snapshot):
    """Project the immutable (non-time, non-material) snapshot inputs the
    proof was minted with, as recorded in its embedded snapshot.

    The deadline-rule structure is recoverable from the snapshot
    schedule (``offset``/``unit``/``depends_on`` per obligation); the
    nested evidence rules are not part of the snapshot output, so a
    structure change with every recoverable field equal is attributed to
    ``rules``.
    """
    obligations = {
        s["obligation"]: {
            "offset": s["offset"],
            "unit": s["unit"],
            "depends_on": list(s["depends_on"]),
        }
        for s in original_snapshot.get("schedule", [])
    }
    return {
        "release_at": original_snapshot["release_at"],
        "timezone": original_snapshot["timezone"],
        "workdays": list(original_snapshot.get("workdays", [])),
        "holidays": list(original_snapshot.get("holidays", [])),
        "obligations": obligations,
    }


def _structural_conflict(batch_id, attestation_id, changed):
    return ConflictError(
        "validity review of attestation %r refuses inputs that differ from"
        " the proof outside the allowed time/material channels: %s"
        % (attestation_id, ", ".join(changed)),
        {
            "reason": "admission-binding-changed",
            "attestation_id": attestation_id,
            "batch_id": batch_id,
            "changed_fields": changed,
        },
    )


def build_validity_review(
    body, attestation, manifest, attestation_id, request_hash
):
    """Recompute the proof's snapshot and assemble the review conclusion.

    Runs with the immutable proof and frozen batch already loaded in one
    read-only transaction; this function itself writes nothing and never
    mutates *attestation*.

    *attestation* is the stored attestation body as minted
    (``batch_id``/``batch_hash``/``snapshot`` embedded) and *request_hash*
    is the normalized-input hash the proof was content-addressed with.
    Bindings are checked first (409), the new snapshot content is then
    parsed (422), and only ``now``/``materials`` may differ from the
    proof -- every other snapshot input must reproduce that request hash
    exactly, which pins release_at, timezone, workdays, holidays,
    deadline rules and evidence rules at once.
    """
    bound_batch_id = attestation.get("batch_id")
    bound_batch_hash = attestation.get("batch_hash")
    original_snapshot = attestation.get("snapshot")
    if not isinstance(original_snapshot, dict) or not bound_batch_hash:
        # Defensive: a row without the embedded proof content cannot be
        # replayed, so no review conclusion may be served from it.
        raise ConflictError(
            "attestation %r cannot be reviewed: its stored proof is incomplete"
            % attestation_id,
            {"reason": "attestation-unverifiable",
             "attestation_id": attestation_id},
        )

    if body["batch_id"] != bound_batch_id:
        raise ConflictError(
            "review batch_id %r is not the batch %r bound to attestation %r"
            % (body["batch_id"], bound_batch_id, attestation_id),
            {
                "reason": "attestation-batch-mismatch",
                "attestation_id": attestation_id,
                "submitted_batch_id": body["batch_id"],
                "bound_batch_id": bound_batch_id,
            },
        )
    if manifest is None or manifest.get("batch_id") != bound_batch_id:
        # Defensive: the store always loads the bound batch; reaching
        # here means the stored foreign key no longer resolves.
        raise ConflictError(
            "the frozen batch %r bound to attestation %r can no longer be"
            " loaded" % (bound_batch_id, attestation_id),
            {"reason": "batch-not-found", "batch_id": bound_batch_id},
        )
    if manifest.get("batch_hash") != bound_batch_hash:
        # Defensive: the frozen manifest moved under its immutable id.
        raise ConflictError(
            "batch %r no longer hashes to the hash %s bound by attestation %r"
            % (bound_batch_id, bound_batch_hash, attestation_id),
            {
                "reason": "batch-hash-mismatch",
                "attestation_id": attestation_id,
                "batch_id": bound_batch_id,
                "expected_batch_hash": bound_batch_hash,
                "registered_batch_hash": manifest.get("batch_hash"),
            },
        )
    if body["batch_hash"] != bound_batch_hash:
        raise ConflictError(
            "submitted batch hash %s does not match the hash %s bound by"
            " attestation %r"
            % (body["batch_hash"], bound_batch_hash, attestation_id),
            {
                "reason": "batch-hash-mismatch",
                "attestation_id": attestation_id,
                "batch_id": bound_batch_id,
                "expected_batch_hash": body["batch_hash"],
                "registered_batch_hash": bound_batch_hash,
            },
        )

    # 1) Recompute the snapshot from the new now/materials.  Any invalid
    #    scheduling or evidence content raises 422 here, before any
    #    structural comparison, so a malformed review never masquerades
    #    as a binding conflict.
    snapshot, normalized = compute_batch_snapshot(
        body, manifest, bound_batch_id
    )

    # 2) Prove the immutable rule structure.  Replay the snapshot with
    #    ONLY the two allowed channels (now, materials) reset to the
    #    proof's minted values; the normalized request hash must then
    #    equal the request_hash the proof was content-addressed with.
    #    Order/UTC-spelling independent, and it pins release_at,
    #    timezone, workdays, holidays, deadline rules and evidence rules
    #    in one comparison.
    replay_body = dict(body)
    replay_body["batch_id"] = bound_batch_id
    replay_body["now"] = original_snapshot["now"]
    replay_body["materials"] = _original_materials(original_snapshot)
    _replay_snapshot, replay_normalized = compute_batch_snapshot(
        replay_body, manifest, bound_batch_id
    )
    if stable_hash(replay_normalized) != request_hash:
        # Pinpoint which immutable structure moved for the error body --
        # derived deterministically from the parsed (valid) submission
        # versus the values embedded in the frozen proof.  The nested
        # evidence rules are not recoverable from the snapshot output, so
        # a conflict with every recoverable field equal is attributed to
        # ``rules``.
        fixed = _original_fixed_inputs(original_snapshot)
        changed = []
        for field in ("release_at", "timezone", "workdays", "holidays",
                      "obligations"):
            if normalized[field] != fixed[field]:
                changed.append(field)
        if not changed:
            changed = ["rules"]
        raise _structural_conflict(bound_batch_id, attestation_id, changed)

    # 3) Replay self-consistency: the stored proof must reproduce itself
    #    (a tampered row can never serve a stable review conclusion).
    if _replay_snapshot["snapshot_hash"] != \
            original_snapshot.get("snapshot_hash"):
        raise ConflictError(
            "attestation %r no longer reproduces its embedded snapshot;"
            " the stored proof cannot be reviewed" % attestation_id,
            {"reason": "attestation-unverifiable",
             "attestation_id": attestation_id},
        )

    # 4) Stable conclusion (overdue dominates, then satisfied-vs-open).
    summary = snapshot["summary"]
    if summary["overdue"] > 0:
        conclusion = "expired"
    elif summary["pending"] > 0:
        conclusion = "pending"
    elif snapshot["snapshot_hash"] == original_snapshot["snapshot_hash"]:
        conclusion = "unchanged"
    else:
        conclusion = "valid-changed"

    # 5) Per-instance differences (status or witness set moved).  Every
    #    instance comes from the frozen manifest, so a missing key would
    #    indicate an internally inconsistent stored proof.
    before = {
        (i["component"], i["obligation"]): i
        for i in original_snapshot.get("instances", [])
    }
    diff_instances = []
    for item in snapshot["instances"]:
        key = (item["component"], item["obligation"])
        old = before.get(key)
        if old is None:
            raise ConflictError(
                "attestation %r embeds no original instance for %r/%r"
                % (attestation_id, item["obligation"], item["component"]),
                {"reason": "attestation-unverifiable",
                 "attestation_id": attestation_id},
            )
        new_witness = list(item.get("satisfied_by", []))
        old_witness = list(old.get("satisfied_by", []))
        if old.get("status") == item["status"] and old_witness == new_witness:
            continue
        diff = {
            "component": item["component"],
            "obligation": item["obligation"],
            "before_status": old.get("status"),
            "after_status": item["status"],
            "deadline": item["deadline"],
            "before_satisfied_by": old_witness,
            "after_satisfied_by": new_witness,
        }
        if item["status"] != "satisfied":
            diff["gaps"] = item.get("gaps", [])
        diff_instances.append(diff)
    diff_instances.sort(key=lambda d: (d["component"], d["obligation"]))

    # 6) Allowed-input movement summary (time + materials only).
    changed_inputs = {
        "now_changed": normalized["now"] != replay_normalized["now"],
        "materials_added": sorted(
            {m["id"] for m in normalized["materials"]}
            - {m["id"] for m in replay_normalized["materials"]}
        ),
        "materials_removed": sorted(
            {m["id"] for m in replay_normalized["materials"]}
            - {m["id"] for m in normalized["materials"]}
        ),
        "materials_changed": sorted(
            m["id"]
            for m in normalized["materials"]
            for o in replay_normalized["materials"]
            if m["id"] == o["id"] and m != o
        ),
    }

    review_content = {
        "attestation_id": attestation_id,
        "batch_id": bound_batch_id,
        "batch_hash": bound_batch_hash,
        "attestation_hash": attestation.get("attestation_hash"),
        "conclusion": conclusion,
        "original_snapshot_hash": original_snapshot.get("snapshot_hash"),
        "snapshot_hash": snapshot["snapshot_hash"],
        "summary": summary,
        "diff_instances": diff_instances,
        "changed_inputs": changed_inputs,
        "reviewed_at": snapshot["now"],
    }
    return {
        "attestation_id": attestation_id,
        "batch_id": bound_batch_id,
        "batch_hash": bound_batch_hash,
        "attestation_hash": attestation.get("attestation_hash"),
        "conclusion": conclusion,
        "original_snapshot_hash": original_snapshot.get("snapshot_hash"),
        "snapshot_hash": snapshot["snapshot_hash"],
        "review_hash": stable_hash(review_content),
        "summary": summary,
        "diff_instances": diff_instances,
        "changed_inputs": changed_inputs,
        "snapshot": snapshot,
    }
