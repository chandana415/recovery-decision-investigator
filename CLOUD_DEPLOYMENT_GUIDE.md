# Streamlit Community Cloud Deployment Guide

## Quick Start (5 Steps)

### 1. Prepare Repository
```bash
# Ensure all changes are committed
git status
git add .
git commit -m "Phase 1-2 complete: upload support + deployment readiness"
git push origin main
```

### 2. Create Streamlit Cloud Account
- Visit: https://streamlit.io/cloud
- Sign in with GitHub
- Authorize Streamlit Cloud access to your repositories

### 3. Connect Your Repository
- Click "New app"
- Select repository: `chandana415/Work`
- Select branch: `main`
- Set main file: `recovery_workspace/app.py`
- Click "Deploy"

### 4. Configure Secrets
After deployment starts:
- In your Streamlit Cloud app dashboard, click "Settings"
- Go to "Secrets"
- Add:
  ```toml
  OPENAI_API_KEY = "sk-your-real-key-here"
  ```
- Save

### 5. Verify Deployment
- Wait 2-3 minutes for initial build
- Visit your app URL
- Test:
  - ✅ Demo scenario (search: `job-weekly-initech-15029`)
  - ✅ Upload logs (upload test .json file)
  - ✅ Live AI mode (if API key is valid)

---

## Files Provided for Cloud Deployment

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies (automatically installed by Cloud) |
| `packages.txt` | System packages (empty; ready for future use) |
| `.streamlit/config.toml` | App configuration (UI theme, server settings) |
| `.env.example` | Local dev template |
| `DEPLOYMENT.md` | Comprehensive deployment checklist |

---

## Environment Variables

### Streamlit Cloud (Secrets Dashboard)
```toml
OPENAI_API_KEY = "sk-proj-your-real-key"
```

### Local Development (.env)
```bash
OPENAI_API_KEY=sk-proj-your-local-key
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **App crashes on startup** | Check Cloud logs; verify `OPENAI_API_KEY` is set in Secrets |
| **"No recovery job found"** | Confirm `mock-data/scenarios/` is in repo; check file paths |
| **Upload fails** | Verify `.log`, `.txt`, or `.json` extension; file < 10 MB |
| **Live AI times out** | Check OpenAI API status; consider increasing timeout |

---

## API Usage Monitoring

To monitor OpenAI API usage:
1. Visit https://platform.openai.com/account/usage/overview
2. Set up usage alerts to prevent unexpected charges
3. Note: Free trial accounts have limits

---

## Security Best Practices

✅ Never commit API keys  
✅ Use Streamlit Cloud Secrets, not environment variables  
✅ Rotate keys regularly  
✅ Monitor usage and set spending limits  
✅ Keep dependencies updated  

---

## Next Steps

- Read `DEPLOYMENT.md` for detailed checklist
- Review `README.md` for project overview
- Check `PHASE2_COMPLETION.md` for deployment readiness summary

---

**Status**: Ready to deploy! 🚀
