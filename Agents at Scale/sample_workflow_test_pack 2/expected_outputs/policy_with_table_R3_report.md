# Access Control Policy — Review Report

## Summary
This document contains **3** requirement statements (2 MUST, 1 SHOULD). The requirements are clear at a high level, but several implementation details are missing (accountability, evidence retention, and exception handling).

## Obligations

| Type | Statement | Section |
|---|---|---|
| MUST | Enforce MFA for all privileged accounts. | Requirements |
| MUST | Review access for admin groups monthly. | Requirements |
| SHOULD | Implement just-in-time access for production systems where feasible. | Requirements |

## Gaps

| Severity | Gap | Related Obligation |
|---|---|---|
| Medium | MFA is required but acceptable methods and exception/approval process are not defined. | Enforce MFA for all privileged accounts. |
| Medium | Monthly review ownership, evidence retention, and escalation for overdue reviews are not defined. | Review access for admin groups monthly. |
| Low | “Feasible” criteria and target scope/timeline for JIT access are not defined. | Implement just-in-time access for production systems where feasible. |

## Questions for Policy Owner
- Who is accountable for performing and attesting monthly admin access reviews (role/team)?
- What evidence must be retained for access reviews and for how long?
- Are any privileged accounts exempt from MFA, and if so what is the approval process?
- What does “feasible” mean for just-in-time access, and is there a target implementation timeline?
