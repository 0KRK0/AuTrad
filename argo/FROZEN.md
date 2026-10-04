# ARGO is frozen (v1.4, 2026-10-04)

ARGO has no demonstrated return edge, and ARGO-Meta was no better than random allocation (v0.7–v0.8). From v1.4 it is
frozen: no new experts, features or model changes. `tests/test_v14.py::test_argo_is_frozen` pins the hash of every
file in this folder.

ARGO's future role is the **risk layer**, after an opportunity has passed the full gate, AEM and a forward paper test:

    information (analyst revisions · Company Memory · NSE depth) -> candidate signal -> economic test -> AEM
        -> risk gate -> paper / forward -> ARGO (risk layer)

Unfreezing needs a new pre-registration that says why, and an update of the pinned hashes in the same change.
