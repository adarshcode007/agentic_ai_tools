# Enterprise Compliance & Retention Policy

## Section 1: Customer Data Isolation
Customer PII must be encrypted at rest using AES-256 and in transit using TLS 1.3.
Multi-tenant databases must enforce tenant ID row-level security (RLS) on all read and write queries.

## Section 2: Disaster Recovery Objectives
The Recovery Point Objective (RPO) is strictly 0 seconds for financial ledger data.
The Recovery Time Objective (RTO) for mission-critical payment services is less than 60 seconds.
