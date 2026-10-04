"""Shared immutable accounting snapshots for persisted assessments.

Confirmation, assessment comparison, remediation planning and the
rules-publication impact review all reason about a *persisted* assessment.
They share one loader (``Store.load_accounting_snapshot`` /
``Store.load_accounting_snapshots``) that returns an
:class:`AccountingSnapshot`: the assessment's normalized request, its
rules snapshot, its persisted result and its revision provenance, read
inside one read-only SQLite transaction and proven consistent *before*
any entry point is allowed to project over it.

A snapshot is accepted only when every correspondence below is proven in
that same transaction:

1. the persisted request is already normalized (re-running
   :func:`solver.normalize_request` reproduces it bit-for-bit);
2. the assessment id is the stable hash of ``{request, rules}``, i.e. the
   persisted request and the pinned rules snapshot really are the pair
   the id was minted from;
3. re-running the assessment with that pair reproduces the persisted
   ``result.summary_hash`` exactly (and the result's summary hash agrees
   with the denormalized ``assessments.summary_hash`` column);
4. when the row names a ruleset revision, the rules snapshot's content
   hash equals that immutable revision row's ``content_hash`` -- the
   snapshot is the revision content, not whatever the ruleset currently
   contains.

Pre-revision rows carry no rules snapshot on disk.  They are accepted
exactly when their original rules can be *recovered* in the transaction
(a same-named stored ruleset whose content hashes back to the
assessment id) and the recovered pair still reproduces the stored
summary hash -- the same rule migration already follows.  Rows that
cannot be restored this way (a deleted/overwritten ruleset, a legacy
inline-rule request, any hash mismatch) are refused outright; no entry
point ever substitutes the currently active ruleset for them.

The snapshot is an immutable value object: every consumer (sorting for
the comparison/impact reports, hash projections, counterfactual and
remediation re-evaluations) reads from this one frozen pair, so one
entry point can never run on historical rules while another silently
falls back to active rules.
"""

from .common import stable_hash
from .solver import assess_normalized, normalize_request


class AccountingSnapshot:
    """One verified, immutable (request, result, rules, provenance) tuple.

    Instances are produced only by the store loader after every
    correspondence check has passed; they are read-only value objects.
    """

    __slots__ = ("id", "ruleset", "revision", "rules", "request", "result")

    def __init__(self, aid, ruleset, revision, rules, request, result):
        object.__setattr__(self, "id", aid)
        object.__setattr__(self, "ruleset", ruleset)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "result", result)

    def as_row(self):
        """Plain-dict projection (the legacy store-row shape).

        The projection is a new dict; its nested request/result/rules
        values are the frozen snapshot contents and must not be mutated.
        """
        return {
            "id": self.id,
            "ruleset": self.ruleset,
            "rules": self.rules,
            "revision": self.revision,
            "request": self.request,
            "result": self.result,
        }


def verify_accounting_tuple(aid, request, result, rules, summary_hash=None):
    """Prove one ``(request, result, rules)`` tuple without any I/O.

    Returns ``True`` exactly when:

    1. the persisted request is already normalized (re-running
       :func:`solver.normalize_request` reproduces it bit-for-bit);
    2. the assessment id binds the exact ``(request, rules)`` pair;
    3. the persisted result body really hashes to the
       ``summary_hash`` it carries (and, when given, to the
       denormalized ``assessments.summary_hash`` column);
    4. re-assessing the pair reproduces that same summary hash.

    Checking the result body's own hash means a persisted result whose
    conclusion was tampered with but kept its old hash field cannot pass.
    Pure content verification shared by the snapshot loader, the legacy
    migration pin and on-demand legacy recovery.
    """
    if rules is None or not isinstance(request, dict) or not isinstance(result, dict):
        return False
    # 1. the persisted request must already be the normalized form
    if normalize_request(request) != request:
        return False
    # 2. the id must bind the exact (request, rules) pair
    if stable_hash({"request": request, "rules": rules})[:32] != aid:
        return False
    # 3. the result body must hash to the summary hash it carries.  The
    #    hash was computed over the body without the self-referential
    #    summary_hash field (see solver._finalize), so drop it first.
    stored_hash = result.get("summary_hash")
    if not isinstance(stored_hash, str) or not stored_hash:
        return False
    body = dict(result)
    body.pop("summary_hash", None)
    if stable_hash(body) != stored_hash:
        return False
    if summary_hash is not None and summary_hash != stored_hash:
        return False
    # 4. replay of the pinned pair must reproduce that conclusion hash
    replayed = assess_normalized(request, rules)
    return replayed.get("summary_hash") == stored_hash
