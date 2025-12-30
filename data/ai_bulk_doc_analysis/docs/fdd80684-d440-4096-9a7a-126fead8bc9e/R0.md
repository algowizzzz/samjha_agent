# ACME Bank — Incident Response SOP (Excerpt)

**Effective Date:** 2025-02-15  
**Owner:** Security Operations

## Objective
Provide a consistent process for triaging and responding to security incidents.

## Procedure
1. **Detect & Triage**
   - Log incident in the case system.
   - Classify severity as *Low / Medium / High / Critical*.
2. **Contain**
   - Isolate affected systems.
   - Preserve evidence.
3. **Eradicate & Recover**
   - Remove malicious artifacts.
   - Restore services and validate.

## Controls Matrix

| Control | Requirement | Evidence |
|---|---|---|
| Logging | MUST retain logs for 365 days | SIEM retention config |
| Access | MUST enforce MFA for admin accounts | IAM policy |
| Backups | SHOULD test restores quarterly | DR test report |

> Note: Images are intentionally omitted for programmatic extraction tests.
