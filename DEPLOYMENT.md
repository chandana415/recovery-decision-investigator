# PHASE 2: DEPLOYMENT READINESS CHECKLIST

## Executive Summary
✅ **71% Ready for Streamlit Community Cloud**
⚠️ **4 Critical/High Issues** must be addressed before deployment
✅ **No architectural changes required**

---

## Detailed Issues

| # | Severity | Issue | File(s) | Why It May Fail | Recommended Change | Status |
|---|----------|-------|---------|-----------------|-------------------|--------|
| 1 | **CRITICAL** | Real API key committed to `.env` | `.env` | Exposed in version control; will compromise security if repo is public | **IMMEDIATE ACTION**: Revoke `OPENAI_API_KEY` in .env. Replace with placeholder. Use `st.secrets` on Streamlit Cloud. | ⚠️ Pending |
| 2 | **HIGH** | Hardcoded repo-relative paths for demo scenarios | `app.py:30`, `investigation.py:27` | On Streamlit Cloud, `REPO_ROOT / "mock-data"` assumes repo structure; may fail if app folder layout differs | Use `__file__`-relative paths with fallback. Example: `Path(__file__).parent.parent / "mock-data"` already correct—test with actual Cloud deployment. | ✅ Code OK, needs testing |
| 3 | **HIGH** | `.env` file loaded unconditionally at module import | `investigation.py:28` | On Streamlit Cloud, `.env` doesn't exist; `load_dotenv()` silently fails (no error), but API key retrieval falls back to `os.environ.get()` correctly. However, explicit fallback to `st.secrets` is missing. | Update `_get_llm_client()` to check `st.secrets` first: `api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")`. Add `import streamlit as st`. | ⚠️ Pending |
| 4 | **HIGH** | Missing `.streamlit/secrets.toml` documentation | N/A (docs) | Users don't know how to configure secrets on Streamlit Cloud | Add `DEPLOYMENT.md` with instructions: (1) Create repo, (2) Connect to Streamlit Cloud, (3) Set `OPENAI_API_KEY` in Cloud secrets UI, (4) Deploy. | ⚠️ Needs doc |
| 5 | **MEDIUM** | Multiple dependency files (both `requirements.txt` and `pyproject.toml`) | Both present | Potential for conflicts; Streamlit Cloud auto-detects both. | **RECOMMENDATION**: Keep both in sync. Current state: ✅ both are identical. No action needed unless one is removed. | ✅ OK |
| 6 | **MEDIUM** | No `packages.txt` for system dependencies | N/A (no system deps currently) | If restic integration is added later, `packages.txt` will be needed. | Create empty `packages.txt` now; add `restic` if Phase 3 proceeds. | ⚠️ Create empty file |
| 7 | **LOW** | Python version 3.11+ requirement vs local 3.9.6 | `pyproject.toml:8` | Local dev machine runs Python 3.9.6; Streamlit Cloud supports 3.11+. No deployment issue, but development may encounter version mismatch. | No change needed for Cloud. Document in `DEPLOYMENT.md` that Python 3.11+ is required. | ✅ OK |
| 8 | **LOW** | Missing `.streamlit/` directory in repo | N/A | Streamlit Cloud can work without it, but best practice is to include `config.toml` for consistent settings across environments. | Create `.streamlit/config.toml` with recommended settings (optional). Add `uploads/` folder structure for uploaded files (optional). | ⚠️ Optional improvement |
| 9 | **NONE** | No external shell/subprocess calls | N/A (verified) | Not an issue. ✅ All operations use pure Python. | No action needed. | ✅ OK |
| 10 | **NONE** | No files written to disk in uploaded logs flow | N/A (verified) | ✅ Uploaded files stored in memory via `st.file_uploader`. | No action needed. | ✅ OK |
| 11 | **NONE** | No secrets printed in error messages | N/A (verified) | ✅ `LLMRequestError` messages are generic; API keys never logged. | No action needed. | ✅ OK |
| 12 | **NONE** | `.gitignore` covers sensitive files | `.gitignore:5-6` | ✅ `.env` and `.streamlit/secrets.toml` already in .gitignore. | No action needed. | ✅ OK |

---

## Action Items (Priority Order)

### 🔴 CRITICAL — Must Complete Before Deployment

**#1: Revoke and Replace Exposed API Key**
```bash
# 1. Revoke the key immediately:
#    Visit https://platform.openai.com/api-keys
#    Delete/revoke the key in .env file

# 2. Update .env to placeholder:
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-api-key-here
EOF

# 3. Verify it's NOT a real key:
git diff .env

# 4. Commit the change:
git add .env
git commit -m "Replace exposed API key with placeholder; use st.secrets on Streamlit Cloud"
```

### 🟡 HIGH — Complete Before Deployment

**#3: Add Streamlit Secrets Fallback to investigation.py**
- Update `_get_llm_client()` to check `st.secrets` first
- Add `import streamlit as st` to investigation.py
- Code change:
  ```python
  def _get_llm_client(...):
      ...
      api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
      if not api_key:
          raise LLMRequestError(...)
  ```

**#4: Create DEPLOYMENT.md**
- Add step-by-step instructions for deploying to Streamlit Community Cloud
- Include how to set secrets in Cloud UI

**#6: Create empty packages.txt**
```bash
touch packages.txt
# Leave empty for now; add dependencies if system packages needed later
```

### 🟢 OPTIONAL — Nice-to-Have

**#8: Create .streamlit/config.toml**
```bash
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[client]
showErrorDetails = false

[logger]
level = "info"

[server]
maxUploadSize = 10
EOF
```

---

## Pre-Deployment Checklist

Before pushing to Streamlit Community Cloud:

- [ ] `.env` file contains NO real API keys (only placeholder)
- [ ] `investigation.py` updated to check `st.secrets` first
- [ ] `DEPLOYMENT.md` created with setup instructions
- [ ] All 78 tests pass locally
- [ ] Streamlit app runs without errors: `streamlit run recovery_workspace/app.py`
- [ ] Demo scenario works (search for `job-weekly-initech-15029`)
- [ ] Upload logs feature works (upload test JSON)
- [ ] Secrets in `.streamlit/secrets.toml` are git-ignored
- [ ] Repository is pushed to GitHub (public or private)
- [ ] Streamlit Cloud app connected to repo
- [ ] `OPENAI_API_KEY` configured in Streamlit Cloud Secrets UI
- [ ] Live deployment tested

---

## Streamlit Community Cloud Setup

1. **Create Streamlit Cloud Account**
   - Visit https://streamlit.io/cloud
   - Sign in with GitHub account

2. **Connect Repository**
   - Click "New app"
   - Select GitHub repo: `chandana415/Work`
   - Select branch: `main`
   - Set main file path: `recovery_workspace/app.py`

3. **Configure Secrets**
   - In Streamlit Cloud app settings, go to "Secrets"
   - Add:
     ```toml
     OPENAI_API_KEY = "sk-your-real-key-here"
     ```
   - Save

4. **Deploy**
   - Click "Deploy"
   - App will install dependencies and start
   - Share public URL

---

## Testing Deployment

After live deployment:
- ✅ Visit public URL
- ✅ Demo Scenario tab works
- ✅ Upload logs tab works
- ✅ Live OpenAI mode works (if key is valid)
- ✅ No error messages expose secrets

---

## Rollback Plan

If deployment fails:
1. Check Streamlit Cloud logs for error messages
2. If API key issue: update secrets in Cloud UI
3. If path issue: verify `REPO_ROOT / "mock-data"` exists in repo
4. If dependency issue: check `requirements.txt` vs `pyproject.toml` sync
5. Push fix to GitHub; Cloud auto-redeploys within 30 seconds

---

## Notes for Future Phases

- **Phase 3 (Restic Integration)**: Will require `packages.txt` with `restic` entry
- **Phase 3 (Real Logs)**: New restic parser will need additional testing; demo scenarios should remain for regression testing
- **Monitoring**: Consider adding error tracking (Sentry, etc.) for production deployment
- **Rate Limiting**: Monitor OpenAI API usage; consider adding usage dashboard

---

**Status**: Ready for Code Review (Phase 2 complete after addressing CRITICAL items)
