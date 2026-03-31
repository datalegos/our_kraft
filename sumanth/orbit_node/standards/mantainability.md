# Steering: Maintainability, Observability, Security, and Operational Rules
Version: 1.0
Owner: NJS — Architecture Team
Purpose: Enforce system-level engineering rules for HRMS code, configuration, outputs, and runbooks.

--- INSTRUCTION SUMMARY ---
You are an automated engineering assistant (Kiro). For every change you generate (code, config, migration, infrastructure manifest, or documentation), follow these mandatory rules and produce the required artifacts and checks in the exact structure and format specified below.

--- REQUIRED PRINCIPLES (non-negotiable) ---
1. Parameter-driven configuration
   - All runtime configuration must be parameterised via environment variables and a single canonical config loader.
   - **Do not** hard-code environment-specific values.
   - Provide a sample `.env.example` and a production-safe `config.template.yaml`.

2. Observability & Metrics
   - All services and worker modules must emit Prometheus-compatible metrics for: request_count, request_latency_ms (histogram), error_count, db_query_time_ms, background_job_runs, background_job_failures.
   - Expose `/metrics` endpoint for Prometheus scraping.
   - All traces must support OpenTelemetry and propagate `trace_id` through the request lifecycle and across background jobs.

3. Alerting & Vulnerability Monitoring
   - CI must run dependency vulnerability scan on every PR. Provide a flagged policy and action plan configuration for alerts (e.g., Slack/email webhook placeholder).
   - Provide an alerts manifest (Prometheus Alertmanager rules) including at minimum: high error-rate, slow queries, job failure spike, and vulnerable-dependency-detected.
   - Define escalation path in `alerts/ESCALATION.md`.

4. Logging standards
   - All logs must be structured JSON with fields at minimum: `timestamp` (ISO 8601 UTC), `level`, `service`, `module`, `trace_id`, `job_id` (if background), `error_code` (if error), `message`, `context` (object).
   - Use RFC3339/ISO-8601 formatting for timestamps (`YYYY-MM-DDThh:mm:ssZ`).
   - Provide a centralized logging configuration snippet (e.g., `logging.config.json`) demonstrating the format and rotation policy.

5. Error identification & traceability
   - Every thrown or logged error must include a unique, stable error code: `{SERVICE}-{MODULE}-{NNNN}` (e.g., `HRMS-AUTH-0001`).
   - Maintain `docs/error-codes.md` with descriptions, probable causes, remediation steps, and severity. New codes must be appended in PRs alongside tests.

6. Database schema changes & backfill strategy
   - Any new DB column addition must include:
     - An Alembic migration file.
     - A backfill plan script (idempotent) that populates values for existing rows with safe defaults or derived values.
     - Estimated runtime and locking considerations documented in `migrations/README.md`.
   - Backfill scripts must be runnable separately from migrations and include a `--dry-run` mode producing a sample report.

7. Functional flow diagrams
   - For every functionality (feature, endpoint, background job), produce a flow diagram that maps the flow down to the function/method name.
   - Use Mermaid syntax and place diagram file under `docs/flows/<feature>.mmd`.
   - Example structure: `API -> Router -> Service.process_<action> -> Repo.insert_<entity>`.

8. Externalized outputs & configs
   - Code folder (read-only run by app): `/opt/hrms`
   - All outputs must be written to: `/opt/hrms_shared_data/outputs`
   - All runtime config must be mounted/read from: `/opt/hrms_shared_data/config`
   - Agent must not write runtime artifacts under `/opt/hrms` except ephemeral logs; production outputs and configs always go to `/opt/hrms_shared_data/*`.

--- OUTPUT REQUIREMENTS FOR EACH PR / TASK ---
For every feature or change, Kiro must produce the following files and place them in the repo (paths relative to repo root):

1. `ops/changes/<ticket-id>/STAGING_README.md`
   - Short summary, files changed, deployment steps, rollback instructions, and estimated impact.

2. `docs/flows/<ticket-id>.mmd`
   - Mermaid flow diagram down to function names.

3. `migrations/<timestamp>_add_<column>.py` (if DB change)
   - Alembic migration file.
   - `migrations/backfill_<timestamp>_add_<column>.py` (idempotent backfill with `--dry-run`).

4. `ops/logging/logging.config.json`
   - Structured logging config sample consistent with logging standard.

5. `ops/metrics/alert_rules.yaml`
   - Prometheus alert rules including vulnerable-deps rule.

6. `configs/config.template.yaml` and `.env.example`
   - Parameterised, documenting required env vars and defaults.

7. `docs/error-codes.md` (append entry for any new error code)

8. Unit & integration tests for added behavior (pytest for backend; include CI test entry).

9. `ops/outputs/` should contain sample run outputs (for the feature) that always map to `/opt/hrms_shared_data/outputs` in deployment.

--- FLOW DIAGRAM TEMPLATE (Mermaid) ---
Use this template for `docs/flows/<feature>.mmd`:

```mermaid
flowchart TD
  A[HTTP POST /leave/apply] --> B[Router: leave_router.apply_leave]
  B --> C[Service: LeaveService.validate_request]
  C --> D[Service: LeaveService.calculate_entitlement]
  D --> E[Repo: LeaveRepo.insert_leave_request]
  E --> F[Background Job: NotificationWorker.enqueue_leave_notification]
  F --> G[Worker: NotificationWorker.send_email]
