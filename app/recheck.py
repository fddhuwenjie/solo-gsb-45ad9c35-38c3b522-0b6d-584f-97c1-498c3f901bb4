"""Continuous validity rechecks for immutable admission attestations.

A registered admission proof is frozen forever, but the world around it
is not: time advances past obligation deadlines and fulfillment
materials get supplemented, corrected or retracted.  This module
re-verifies one registered attestation against a *new* set of snapshot
inputs without ever touching the stored proof.

The caller re-declares the attestation's binding -- ``batch_id``,
``batch_hash`` and ``release_at`` -- together with the full snapshot
input set (a new ``now``, workdays/holidays, the deadline rules and the
materials/evidence rules).  Inside one read-only SQLite transaction the
service loads the immutable proof and its frozen batch manifest,
requires every non-time/non-material input to match the proof's
registered content (batch hash, release instant, timezone,
workdays/holidays, deadline rules and evidence rules -- only ``now``
and ``materials`` may drift), recomputes the fulfillment snapshot and
reports a stable conclusion:

* ``unchanged``     -- the recomputed snapshot is identical to the
  proof's embedded snapshot (same instant, same materials);
* ``valid-changed`` -- inputs changed but every obligation instance is
  still ``satisfied``;
* ``pending``       -- at least one instance is unsatisfied but its
  deadline has not passed yet;
* ``expired``       -- at least one instance is ``overdue``.

The response carries the per-instance differences against the proof's
snapshot and a ``review_hash`` over the canonical review; request
ordering (materials, object keys, workdays/holidays/depends_on) and
equivalent RFC 3339 spellings never change either.  The recheck is
strictly read-only: the registered attestation is never rewritten.

An unknown attestation yields 404; a batch id/hash or rule-structure
binding conflict yields 409; invalid scheduling/material/evidence
content yields 422, exactly as on the stateless snapshot endpoint.
"""

import re

from .common import ClientError, ConflictError, stable_hash
from .snapshot import compute_batch_snapshot

_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# Snapshot fields whose presence the recheck envelope asserts; their
# content is validated by the snapshot builder (HTTP 422), exactly as on
# the stateless snapshot endpoint.
_SNAPSHOT_FIELDS = (
    "now",
    "release_at",
    "timezone",
    "workdays",
    "obligations",
)


def validate_recheck_body(body):
    """Validate the recheck request envelope (HTTP 400 rules only).

    Returns ``(batch_id, expected_batch_hash, snapshot_body)`` where
    *snapshot_body* is the original payload handed to the snapshot
    builder inside the store's read transaction; every
    scheduling/evidence/content error surfaces there as HTTP 422.
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
            " registered batch hash"
        )
    expected_hash = expected_hash.strip()

    for field in _SNAPSHOT_FIELDS:
        if field not in body:
            raise ClientError("recheck request requires %r" % field)
    if "holidays" not in body:
        raise ClientError("recheck request requires 'holidays'")

    return batch_id, expected_hash, body


def build_validity_recheck(
    attestation,
    request_hash,
    manifest,
    attestation_id,
    batch_id,
    expected_hash,
    body,
):
    """Recompute the proof's snapshot for new inputs and classify it.

    Runs inside the store's open read transaction, *after* the caller
    has confirmed the attestation exists (404 otherwise).  The submitted
    batch binding and every non-time/non-material snapshot input must
    match the registered proof; only ``now`` and ``materials`` may
    differ.  Returns the review response including ``review_hash``.
    """
    bound_batch_id = attestation["batch_id"]
    if batch_id != bound_batch_id:
        raise ConflictError(
            "recheck names batch %r but attestation %r is bound to batch"
            " %r" % (batch_id, attestation_id, bound_batch_id),
            {
                "reason": "batch-id-mismatch",
                "attestation_id": attestation_id,
                "expected_batch_id": batch_id,
                "registered_batch_id": bound_batch_id,
            },
        )

    bound_hash = attestation["batch_hash"]
    if expected_hash != bound_hash:
        raise ConflictError(
            "expected batch hash %s does not match the hash %s bound to"
            " attestation %r" % (expected_hash, bound_hash, attestation_id),
            {
                "reason": "batch-hash-mismatch",
                "attestation_id": attestation_id,
                "batch_id": bound_batch_id,
                "expected_batch_hash": expected_hash,
                "registered_batch_hash": bound_hash,
            },
        )
    if manifest.get("batch_hash") != bound_hash:
        # Unreachable while batches stay immutable; kept as an integrity
        # guard so a corrupted manifest is never rechecked silently.
        raise ConflictError(
            "the frozen manifest of batch %r no longer matches the hash"
            " bound to attestation %r" % (bound_batch_id, attestation_id),
            {
                "reason": "batch-hash-mismatch",
                "attestation_id": attestation_id,
                "batch_id": bound_batch_id,
                "expected_batch_hash": bound_hash,
                "registered_batch_hash": manifest.get("batch_hash"),
            },
        )

    # Any invalid scheduling/material/evidence input raises
    # UnprocessableError (422) here -- before the binding check --
    # exactly as on the stateless snapshot endpoint.
    snapshot, normalized_inputs = compute_batch_snapshot(
        body, manifest, bound_batch_id
    )

    att_snapshot = attestation["snapshot"]
    _check_binding(
        attestation_id, bound_batch_id, att_snapshot, request_hash,
        normalized_inputs,
    )

    changed = _changed_instances(att_snapshot, snapshot)
    review = {
        "attestation_id": attestation_id,
        "batch_id": bound_batch_id,
        "batch_hash": bound_hash,
        "attestation_hash": attestation["attestation_hash"],
        "attestation_snapshot_hash": attestation["snapshot_hash"],
        "attestation_now": att_snapshot["now"],
        "attestation_summary": att_snapshot["summary"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "now": snapshot["now"],
        "conclusion": _classify(att_snapshot, snapshot),
        "summary": snapshot["summary"],
        "changed_instances": changed,
        "changed_count": len(changed),
        "snapshot": snapshot,
    }
    review["review_hash"] = stable_hash(review)
    return review


def _check_binding(
    attestation_id, batch_id, att_snapshot, request_hash, normalized_inputs
):
    """Enforce that only ``now`` and ``materials`` drifted from the proof.

    The attestation row stores the SHA-256 of its normalized snapshot
    inputs.  Substituting the proof's original ``now``/``materials``
    into the recheck's normalized inputs must reproduce that hash
    exactly -- it pins the release instant, timezone, workdays,
    holidays, deadline rules and evidence rules.  Every aspect the
    embedded snapshot echoes is also compared directly so the 409
    response can name the drifting fields.
    """
    fields = []
    for field in ("release_at", "timezone", "workdays", "holidays"):
        if normalized_inputs[field] != att_snapshot[field]:
            fields.append(field)
    original_obligations = {
        entry["obligation"]: {
            "offset": entry["offset"],
            "unit": entry["unit"],
            "depends_on": entry["depends_on"],
        }
        for entry in att_snapshot["schedule"]
    }
    if normalized_inputs["obligations"] != original_obligations:
        fields.append("obligations")

    reconstructed = dict(normalized_inputs)
    reconstructed["now"] = att_snapshot["now"]
    reconstructed["materials"] = att_snapshot["materials"]
    if stable_hash(reconstructed) != request_hash:
        if not fields:
            # Every recoverable aspect matches, so the drift must be in
            # the evidence rules -- the only pinned input the embedded
            # snapshot does not echo.
            fields.append("rules")
        raise ConflictError(
            "recheck inputs drifted from attestation %r in: %s -- only"
            " 'now' and 'materials' may change"
            % (attestation_id, ", ".join(sorted(fields))),
            {
                "reason": "attestation-binding-mismatch",
                "attestation_id": attestation_id,
                "batch_id": batch_id,
                "fields": sorted(fields),
            },
        )


def _classify(att_snapshot, snapshot):
    """Stable conclusion: unchanged > expired > pending > valid-changed."""
    if snapshot["snapshot_hash"] == att_snapshot["snapshot_hash"]:
        return "unchanged"
    summary = snapshot["summary"]
    if summary["overdue"] > 0:
        return "expired"
    if summary["pending"] > 0:
        return "pending"
    return "valid-changed"


def _changed_instances(att_snapshot, snapshot):
    """Per-instance differences between the proof's and the new snapshot.

    Both snapshots derive from the same frozen manifest, so instance
    keys ``(component, obligation)`` align one-to-one; the comparison
    covers the status, the satisfying materials and the gap analysis
    (deadlines are pinned by the binding check and can never drift).
    The proof's instances are iterated in the new snapshot's sorted
    order.
    """
    old = {
        (i["component"], i["obligation"]): i
        for i in att_snapshot["instances"]
    }
    changed = []
    for inst in snapshot["instances"]:
        prev = old.get((inst["component"], inst["obligation"]))
        if prev is None:
            continue
        if (
            prev["status"] == inst["status"]
            and prev["satisfied_by"] == inst["satisfied_by"]
            and prev["gaps"] == inst["gaps"]
        ):
            continue
        changed.append(
            {
                "component": inst["component"],
                "obligation": inst["obligation"],
                "deadline": inst["deadline"],
                "old_status": prev["status"],
                "new_status": inst["status"],
                "old_satisfied_by": prev["satisfied_by"],
                "new_satisfied_by": inst["satisfied_by"],
                "gaps": inst["gaps"],
            }
        )
    return changed
