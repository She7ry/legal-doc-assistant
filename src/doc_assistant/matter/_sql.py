"""SQL DDL/DML 常量 —— MatterStore 所有 SQL 语句集中管理。"""

# ── DDL: matters ──────────────────────────────────────────────

CREATE_TABLE_MATTERS = """
CREATE TABLE IF NOT EXISTS matters (
    matter_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    matter_profile_json TEXT NOT NULL,
    source_task_id TEXT NOT NULL,
    latest_task_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(matter_id, tenant_id, user_id)
)
"""

CREATE_TABLE_MATTER_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS matter_artifacts (
    artifact_id TEXT NOT NULL,
    matter_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    items_json TEXT NOT NULL,
    source_finding_ids_json TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_task_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(artifact_id, matter_id, tenant_id, user_id),
    FOREIGN KEY(matter_id, tenant_id, user_id)
        REFERENCES matters(matter_id, tenant_id, user_id)
)
"""

CREATE_TABLE_REVIEW_FINDINGS = """
CREATE TABLE IF NOT EXISTS review_findings (
    finding_id TEXT NOT NULL,
    matter_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    source_step_id TEXT NOT NULL,
    clause_reference TEXT NOT NULL,
    evidence_coverage TEXT NOT NULL,
    support_level TEXT NOT NULL,
    unsupported_reason TEXT NOT NULL,
    source_quote TEXT NOT NULL,
    location_label TEXT NOT NULL,
    needs_human_review INTEGER NOT NULL,
    human_review_status TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_task_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(finding_id, matter_id, tenant_id, user_id),
    FOREIGN KEY(matter_id, tenant_id, user_id)
        REFERENCES matters(matter_id, tenant_id, user_id)
)
"""

CREATE_TABLE_MATTER_EVENTS = """
CREATE TABLE IF NOT EXISTS matter_events (
    event_id TEXT PRIMARY KEY,
    matter_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    old_value_json TEXT,
    new_value_json TEXT,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(matter_id, tenant_id, user_id)
        REFERENCES matters(matter_id, tenant_id, user_id)
)
"""

CREATE_TABLE_ARTIFACT_VERSIONS = """
CREATE TABLE IF NOT EXISTS matter_artifact_versions (
    artifact_id TEXT NOT NULL,
    matter_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    items_json TEXT NOT NULL,
    source_finding_ids_json TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_task_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(artifact_id, matter_id, tenant_id, user_id, version),
    FOREIGN KEY(matter_id, tenant_id, user_id)
        REFERENCES matters(matter_id, tenant_id, user_id)
)
"""

# ── DDL: indexes ──────────────────────────────────────────────

CREATE_INDEX_MATTERS_USER_UPDATED = """
CREATE INDEX IF NOT EXISTS idx_matters_user_updated
ON matters(tenant_id, user_id, updated_at)
"""

CREATE_INDEX_MATTER_ARTIFACTS_MATTER = """
CREATE INDEX IF NOT EXISTS idx_matter_artifacts_matter
ON matter_artifacts(tenant_id, user_id, matter_id)
"""

CREATE_INDEX_REVIEW_FINDINGS_MATTER = """
CREATE INDEX IF NOT EXISTS idx_review_findings_matter
ON review_findings(tenant_id, user_id, matter_id)
"""

CREATE_INDEX_MATTER_EVENTS_MATTER = """
CREATE INDEX IF NOT EXISTS idx_matter_events_matter
ON matter_events(tenant_id, user_id, matter_id, created_at)
"""

CREATE_INDEX_ARTIFACT_VERSIONS_ARTIFACT = """
CREATE INDEX IF NOT EXISTS idx_artifact_versions_artifact
ON matter_artifact_versions(tenant_id, user_id, matter_id, artifact_id, version)
"""

# ── DML: matters 表 ───────────────────────────────────────────

SELECT_MATTER_BY_IDS = """
SELECT * FROM matters
WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
"""

UPSERT_MATTER = """
INSERT INTO matters (
    matter_id, tenant_id, user_id, title, status, matter_profile_json,
    source_task_id, latest_task_id, created_at, updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(matter_id, tenant_id, user_id)
DO UPDATE SET
    title = excluded.title,
    status = excluded.status,
    matter_profile_json = excluded.matter_profile_json,
    latest_task_id = excluded.latest_task_id,
    updated_at = excluded.updated_at
"""

SELECT_MATTERS_BY_USER = """
SELECT * FROM matters
WHERE tenant_id = ? AND user_id = ?
ORDER BY updated_at DESC
LIMIT ?
"""

UPDATE_MATTER_UPDATED_AT = """
UPDATE matters
SET updated_at = ?
WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
"""

UPDATE_MATTER_STATUS_AND_PROFILE = """
UPDATE matters
SET status = ?, matter_profile_json = ?, updated_at = ?
WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
"""

# ── DML: matter_artifacts 表 ──────────────────────────────────

UPSERT_ARTIFACT = """
INSERT INTO matter_artifacts (
    artifact_id, matter_id, tenant_id, user_id, artifact_type, title, summary,
    items_json, source_finding_ids_json, citations_json, metadata_json,
    source_task_id, version, status, created_at, updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(artifact_id, matter_id, tenant_id, user_id)
DO UPDATE SET
    artifact_type = excluded.artifact_type,
    title = excluded.title,
    summary = excluded.summary,
    items_json = excluded.items_json,
    source_finding_ids_json = excluded.source_finding_ids_json,
    citations_json = excluded.citations_json,
    metadata_json = excluded.metadata_json,
    source_task_id = excluded.source_task_id,
    version = excluded.version,
    status = excluded.status,
    updated_at = excluded.updated_at
"""

SELECT_ARTIFACT_BY_ID = """
SELECT * FROM matter_artifacts
WHERE matter_id = ? AND tenant_id = ? AND user_id = ? AND artifact_id = ?
"""

UPDATE_ARTIFACT = """
UPDATE matter_artifacts
SET title = ?, summary = ?, items_json = ?, metadata_json = ?,
    version = ?, status = ?, updated_at = ?
WHERE matter_id = ? AND tenant_id = ? AND user_id = ? AND artifact_id = ?
"""

SELECT_ARTIFACTS_BY_MATTER = """
SELECT * FROM matter_artifacts
WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
ORDER BY artifact_type ASC
"""

INSERT_CURRENT_ARTIFACT_VERSION = """
INSERT OR REPLACE INTO matter_artifact_versions (
    artifact_id, matter_id, tenant_id, user_id, artifact_type, title, summary,
    items_json, source_finding_ids_json, citations_json, metadata_json,
    source_task_id, version, status, created_at, updated_at
)
SELECT
    artifact_id, matter_id, tenant_id, user_id, artifact_type, title, summary,
    items_json, source_finding_ids_json, citations_json, metadata_json,
    source_task_id, version, status, created_at, updated_at
FROM matter_artifacts
WHERE matter_id = ? AND tenant_id = ? AND user_id = ? AND artifact_id = ?
"""

SELECT_ARTIFACT_VERSIONS = """
SELECT * FROM matter_artifact_versions
WHERE matter_id = ? AND tenant_id = ? AND user_id = ? AND artifact_id = ?
ORDER BY version ASC
"""

BACKFILL_ARTIFACT_VERSIONS = """
INSERT OR IGNORE INTO matter_artifact_versions (
    artifact_id, matter_id, tenant_id, user_id, artifact_type, title, summary,
    items_json, source_finding_ids_json, citations_json, metadata_json,
    source_task_id, version, status, created_at, updated_at
)
SELECT
    artifact_id, matter_id, tenant_id, user_id, artifact_type, title, summary,
    items_json, source_finding_ids_json, citations_json, metadata_json,
    source_task_id, version, status, created_at, updated_at
FROM matter_artifacts
"""

# ── DML: review_findings 表 ───────────────────────────────────

UPSERT_FINDING = """
INSERT INTO review_findings (
    finding_id, matter_id, tenant_id, user_id, category, severity, summary,
    recommended_action, citations_json, source_step_id, clause_reference,
    evidence_coverage, support_level, unsupported_reason, source_quote,
    location_label, needs_human_review, human_review_status, status,
    metadata_json, source_task_id, created_at, updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(finding_id, matter_id, tenant_id, user_id)
DO UPDATE SET
    category = excluded.category,
    severity = excluded.severity,
    summary = excluded.summary,
    recommended_action = excluded.recommended_action,
    citations_json = excluded.citations_json,
    source_step_id = excluded.source_step_id,
    clause_reference = excluded.clause_reference,
    evidence_coverage = excluded.evidence_coverage,
    support_level = excluded.support_level,
    unsupported_reason = excluded.unsupported_reason,
    source_quote = excluded.source_quote,
    location_label = excluded.location_label,
    needs_human_review = excluded.needs_human_review,
    human_review_status = excluded.human_review_status,
    status = excluded.status,
    metadata_json = excluded.metadata_json,
    source_task_id = excluded.source_task_id,
    updated_at = excluded.updated_at
"""

SELECT_FINDING_ROW_BY_ID = """
SELECT * FROM review_findings
WHERE tenant_id = ? AND user_id = ? AND matter_id = ? AND finding_id = ?
"""

UPDATE_FINDING_REVIEW = """
UPDATE review_findings
SET human_review_status = ?, status = ?, metadata_json = ?, updated_at = ?
WHERE tenant_id = ? AND user_id = ? AND matter_id = ? AND finding_id = ?
"""

SELECT_FINDINGS_BY_MATTER = """
SELECT * FROM review_findings
WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
ORDER BY finding_id ASC
"""

# ── DML: matter_events 表 ─────────────────────────────────────

INSERT_EVENT = """
INSERT INTO matter_events (
    event_id, matter_id, tenant_id, user_id, event_type, entity_type,
    entity_id, old_value_json, new_value_json, actor, created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SELECT_EVENTS_BY_MATTER = """
SELECT * FROM matter_events
WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
ORDER BY created_at DESC
LIMIT ?
"""


SCHEMA_STATEMENTS = (
    CREATE_TABLE_MATTERS,
    CREATE_TABLE_MATTER_ARTIFACTS,
    CREATE_TABLE_REVIEW_FINDINGS,
    CREATE_TABLE_MATTER_EVENTS,
    CREATE_TABLE_ARTIFACT_VERSIONS,
    CREATE_INDEX_MATTERS_USER_UPDATED,
    CREATE_INDEX_MATTER_ARTIFACTS_MATTER,
    CREATE_INDEX_REVIEW_FINDINGS_MATTER,
    CREATE_INDEX_MATTER_EVENTS_MATTER,
    CREATE_INDEX_ARTIFACT_VERSIONS_ARTIFACT,
)
