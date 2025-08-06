# Security Guidelines

## 🔐 Environment Variables

### ⚠️ CRITICAL: Never Commit .env Files

**NEVER commit `.env` files to version control.** These files contain sensitive information like:
- API keys
- Database credentials
- Secret keys
- Authentication tokens

### ✅ Proper Setup

1. **Use the template**: Copy `backend/.env.template` to `backend/.env`
2. **Generate secure keys**:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
3. **Fill in your values**: Replace all `CHANGE_THIS_*` placeholders
4. **Verify .gitignore**: Ensure `.env` files are ignored

### 🛡️ Security Best Practices

#### For Development:
- Use strong, unique passwords
- Rotate API keys regularly
- Never share .env files in chat, email, or screenshots
- Use environment variables for all sensitive data

#### For Production:
- Use a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
- Enable SSL/HTTPS
- Set `ENVIRONMENT=production`
- Use PostgreSQL with SSL connections
- Enable security headers

### 🚨 If You Accidentally Commit Secrets

1. **Immediately rotate all exposed keys**
2. **Remove from git history**:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch backend/.env' \
   --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** (⚠️ dangerous - coordinate with team)
4. **Update all keys in production**

### 📝 Environment Variables Reference

Required variables:
- `SECRET_KEY` - Application secret key
- `JWT_SECRET_KEY` - JWT signing key
- `DATABASE_URL` - Database connection string

Optional but recommended:
- `OPENAI_API_KEY` - For AI features
- `POLYGON_API_KEY` - For market data
- `ALPHA_VANTAGE_API_KEY` - Alternative market data
- `REDIS_URL` - For caching and sessions

See `backend/.env.template` for complete list.

---

**Remember**: Security is everyone's responsibility. When in doubt, ask!