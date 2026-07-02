---
name: secret-scan
description: Scan the codebase for hardcoded secrets, API keys, tokens, and credentials. Use before committing or when the user wants to check for credential leaks.
user-invocable: true
---

# Secret Scan

Scan the project for accidentally committed secrets, credentials, and API keys.

## Scan Targets

### High-priority patterns (block commit)

```
# API keys in source code
api_key\s*=\s*["'][A-Za-z0-9_\-]{20,}["']
API_KEY\s*=\s*["'][A-Za-z0-9_\-]{20,}["']
Authorization\s*:\s*Bearer\s+[A-Za-z0-9_\-.]{20,}
x-api-key\s*:\s*[A-Za-z0-9_\-]{20,}

# Private keys
-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----
-----BEGIN PGP PRIVATE KEY BLOCK-----

# Tokens
ghp_[A-Za-z0-9]{36}           # GitHub personal access token
gho_[A-Za-z0-9]{36}           # GitHub OAuth token
xox[bpras]-[A-Za-z0-9]{10,}   # Slack token
sk-[A-Za-z0-9]{32,}           # OpenAI/Anthropic key pattern
AKIA[0-9A-Z]{16}              # AWS Access Key
```

### Medium-priority patterns (warn)

```
# Connection strings with credentials
mongodb\+srv://[^/]+:[^@]+@
postgresql://[^/]+:[^@]+@
mysql://[^/]+:[^@]+@

# Hardcoded passwords
password\s*=\s*["'][^"']{4,}["']
passwd\s*=\s*["'][^"']{4,}["']
```

## Process

1. Run grep for high-priority patterns:

```bash
grep -rInE '(api_key|API_KEY|secret|token|password|passwd)\s*=\s*["'"'"'][A-Za-z0-9_\-]{8,}' \
  --include='*.py' --include='*.js' --include='*.ts' --include='*.vue' \
  --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=__pycache__ \
  --exclude='*.sqlite3' --exclude='*.env.example' .
```

2. Check `.env` and `.env.example` difference:

```bash
diff <(grep -v '^#' .env.example | cut -d= -f1 | sort) <(grep -v '^#' .env 2>/dev/null | grep -v '^$' | cut -d= -f1 | sort)
```

3. Check git history for previously committed secrets:

```bash
git log --all --full-history -p -- '*.env' '.env' 2>/dev/null | grep -E '^\+[^+].*=.*[A-Za-z0-9_\-]{20,}' | head -20
```

4. Verify `.gitignore` covers `.env` and data files:

```bash
grep -E '\.env$|\.sqlite3$|vector_store|uploads' .gitignore
```

## Red Flags

- Any actual secret value (not placeholder like `<your-key>`) in tracked files → **BLOCK COMMIT**
- `.env` file tracked by git → **CRITICAL** (rotate all keys)
- Secrets in git history → **HIGH** (rotate affected keys, consider `git filter-branch` or `BFG Repo-Cleaner`)
