---
name: owasp-review
description: Review API endpoints and backend code against the OWASP API Security Top 10. Use when the user wants a security review of API code, routes, or authentication logic.
user-invocable: true
---

# OWASP API Security Review

Systematically review FastAPI backend code against the OWASP API Security Top 10 (2023).

## Review Checklist

### API1: Broken Object Level Authorization (BOLA)
- Check every endpoint that accepts object IDs: does it verify the requester owns that object?
- Look for missing tenant/user validation in `api/routers/*.py`
- Pattern to find: `def get_xxx(id: str)` without ownership check

### API2: Broken Authentication
- Verify API key validation in `api/dependencies.py` is not bypassable
- Check for hardcoded keys or test-only auth bypass paths
- Verify rate limiting applies to auth endpoints

### API3: Broken Object Property Level Authorization
- Check PATCH/PUT endpoints: do they validate which fields the user can modify?
- Look for mass assignment vulnerabilities in request schemas

### API4: Unrestricted Resource Consumption
- Check file upload limits (`DOC_ASSISTANT_MAX_UPLOAD_BYTES`)
- Verify pagination exists on list endpoints
- Check for unbounded query results

### API5: Broken Function Level Authorization
- Verify admin-only endpoints are actually protected
- Check that `X-Tenant-Id` header is validated (regex: `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`)

### API6: Unrestricted Access to Sensitive Business Flows
- Check for rate limiting on sensitive operations (ingest, agent tasks)

### API7: Server-Side Request Forgery (SSRF)
- Check `tools/web_search.py` — does the web search client validate target URLs?
- Check any URL-fetching endpoints

### API8: Security Misconfiguration
- Verify CORS origins are restrictive, not wildcard
- Check for debug mode in production
- Verify error responses don't leak stack traces (already handled in `api/main.py`)

### API9: Improper Inventory Management
- Check for unversioned/undocumented endpoints
- Verify all routers are intentional

### API10: Unsafe Consumption of APIs
- Check LLM API calls: are prompts sanitized? Is user input separated from system instructions?
- Verify web search results are sanitized before display

## Output Format

For each finding, report:
- **OWASP Category**: API1-API10
- **Severity**: Critical / High / Medium / Low
- **Location**: exact file:line
- **Description**: what's wrong
- **Fix**: concrete remediation
