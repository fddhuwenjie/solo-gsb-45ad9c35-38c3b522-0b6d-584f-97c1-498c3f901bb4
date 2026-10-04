"""Immutable admission attestations for frozen release batches.

An attestation registers an *immutable admission proof* for one release
batch.  The caller submits a unique ``attestation_id``, the ``batch_id``
and the expected ``batch_hash``, together with the exact inputs the batch
fulfillment snapshot endpoint takes: the current and release moments,
timezone, workdays/holidays, per-obligation deadline rules and timed
fulfillment materials/evidence rules.

The store loads the frozen batch inside one ``BEGIN IMMEDIATE`` SQLite
write transaction, verifies the ``batch_hash`` still binds to that batch,
and recomputes the fulfillment snapshot **there**.  The attestation is
written only when every component obligation instance is ``satisfied`` --
a single ``pending`` or ``overdue`` instance, a hash mismatch or any
invalid snapshot input refuses the whole operation and leaves no record.

Attestations are immutable and content-addressed: the same
``attestation_id`` replays idempotently only when the normalized request
content is identical (entry order and equivalent timestamp spellings
never matter); different content under the same id conflicts (409).
Later rule publications or material changes never touch a registered
proof -- it reads only the frozen batch manifest and persists the exact
snapshot, rule revisions and hashes it was minted with.
"""

import re

from .common import ClientError, ConflictError, stable_hash
from .fulfillment import UnprocessableError
from .snapshot import compute_batch_snapshot

_ATTESTATION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# Snapshot fields whose validation and canonicalization are delegated to
# the snapshot builder; the attestation envelope only asserts they exist.
_SNAPSHOT_FIELDS = (
    "now",
    "release_at",
    "timezone",
    "workdays",
    "obligations",
)


def validate_attestation_body(body, path_batch_id=None):
    """Validate the attestation request envelope (HTTP 400 rules only).

    Returns ``(attestation_id, batch_id, expected_batch_hash, snapshot_body)``
    where *snapshot_body* is the original (validated-for-shape) payload
    handed to the snapshot builder inside the store transaction; every
    scheduling/evidence/content error surfaces there as HTTP 422, exactly
    as it does on the stateless snapshot endpoint.  When the route already
    names a batch, the body ``batch_id`` must agree with it (422).
    """
    if not isinstance(body, dict):
        raise ClientError("request body must be a JSON object")

    attestation_id = body.get("attestation_id")
    if not isinstance(attestation_id, str) or not attestation_id.strip():
        raise ClientError("attestation_id must be a non-empty string")
    attestation_id = attestation_id.strip()
    if not _ATTESTATION_ID_RE.match(attestation_id):
        raise ClientError(
            "attestation_id may contain only letters, digits, dot,"
            " underscore and hyphen (1..128 characters)"
        )

    batch_id = body.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ClientError("batch_id must be a non-empty string")
    batch_id = batch_id.strip()
    if not _BATCH_ID_RE.match(batch_id):
        raise ClientError(
            "batch_id may contain only letters, digits, dot, underscore"
            " and hyphen (1..128 characters)"
        )
    if path_batch_id is not None and batch_id != path_batch_id:
        raise UnprocessableError(
            "body batch_id %r does not match path batch_id %r"
            % (batch_id, path_batch_id)
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
            raise ClientError("attestation request requires %r" % field)
    if "holidays" not in body:
        raise ClientError("attestation request requires 'holidays'")

    return attestation_id, batch_id, expected_hash, body


def build_admission_attestation(body, manifest, attestation_id, expected_hash):
    """Recompute the snapshot and assemble the immutable admission proof.

    Runs inside the store's open write transaction, *after* the caller has
    confirmed the batch exists (404 otherwise).  The expected hash is
    compared to the frozen manifest's ``batch_hash``; any mismatch or
    invalid snapshot content raises before a proof is assembled.  When one
    or more obligation instances are not ``satisfied`` the admission is
    refused with the offending instances attached.

    Returns ``(attestation, request_hash)``.
    """
    batch_id = manifest["batch_id"]
    actual_hash = manifest.get("batch_hash")
    if actual_hash != expected_hash:
        raise ConflictError(
            "expected batch hash %s does not match the frozen manifest"
            " hash %s of batch %r" % (expected_hash, actual_hash, batch_id),
            {
                "reason": "batch-hash-mismatch",
                "batch_id": batch_id,
                "expected_batch_hash": expected_hash,
                "registered_batch_hash": actual_hash,
            },
        )

    # Any invalid scheduling/material/evidence input raises UnprocessableError
    # (422) here -- before the admission gate -- leaving no record.
    snapshot, normalized_inputs = compute_batch_snapshot(body, manifest, batch_id)
    summary = snapshot["summary"]
    unsatisfied = [
        {
            "component": item["component"],
            "obligation": item["obligation"],
            "deadline": item["deadline"],
            "status": item["status"],
            "gaps": item["gaps"],
        }
        for item in snapshot["instances"]
        if item["status"] != "satisfied"
    ]
    if unsatisfied:
        raise ConflictError(
            "admission refused for batch %r: %d of %d component obligation"
            " instances are not satisfied"
            % (batch_id, summary["total"] - summary["satisfied"],
               summary["total"]),
            {
                "reason": "obligations-not-satisfied",
                "batch_id": batch_id,
                "snapshot_hash": snapshot["snapshot_hash"],
                "summary": summary,
                "unsatisfied_instances": unsatisfied,
            },
        )

    request_hash = stable_hash(normalized_inputs)
    rule_revisions = sorted(
        (dict(rev) for rev in manifest.get("rule_revisions", [])),
        key=lambda r: r["assessment_id"],
    )
    content = {
        "attestation_id": attestation_id,
        "batch_id": batch_id,
        "batch_hash": actual_hash,
        "snapshot_hash": snapshot["snapshot_hash"],
        "rule_revisions": rule_revisions,
        "snapshot": snapshot,
    }
    attestation = dict(content)
    attestation["attestation_hash"] = stable_hash(content)
    attestation["created"] = True
    return attestation, request_hash
