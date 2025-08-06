#!/usr/bin/env python3
"""
Security Check Script - NEVER commit API keys to GitHub!
This script helps ensure your API keys are properly secured
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SecurityChecker:
    """Comprehensive security checker for API keys and sensitive data"""
    
    # Patterns that indicate potential API keys
    API_KEY_PATTERNS = [
        r'[A-Za-z0-9]{32,}',  # Generic long strings
        r'sk-[A-Za-z0-9]{48}',  # OpenAI keys
        r'Bearer\s+[A-Za-z0-9\-._~+/]+=*',  # Bearer tokens
        r'[A-Za-z0-9]{20,}:[A-Za-z0-9]{20,}',  # Key:secret pairs
    ]
    
    # Known sensitive environment variables
    SENSITIVE_VARS = [
        'API_KEY', 'SECRET', 'PASSWORD', 'TOKEN', 'CREDENTIAL',
        'PRIVATE', 'AUTH', 'ACCESS', 'REFRESH'
    ]
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
    def check_env_file_safety(self) -> bool:
        """Check if .env file is properly gitignored"""
        gitignore_path = Path('.gitignore')
        if not gitignore_path.exists():
            self.issues.append("❌ No .gitignore file found!")
            return False
        
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
            
        # Check for .env patterns
        env_patterns = ['.env', '*.env', '.env.*', '.env*']
        found_patterns = [p for p in env_patterns if p in gitignore_content]
        
        if not found_patterns:
            self.issues.append("❌ .env files are NOT in .gitignore!")
            return False
        else:
            print(f"✅ .env files are properly gitignored: {found_patterns}")
            return True
    
    def scan_for_exposed_keys(self, directory: str = '.') -> List[Tuple[str, int, str]]:
        """Scan source code for hardcoded API keys"""
        exposed = []
        
        # Extensions to scan
        extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yml', '.yaml']
        
        for root, dirs, files in os.walk(directory):
            # Skip directories
            if 'node_modules' in root or '.git' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    if file == '.env' or file.endswith('.env'):
                        continue  # Skip env files themselves
                        
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                # Check for hardcoded keys
                                for pattern in self.API_KEY_PATTERNS:
                                    if re.search(pattern, line):
                                        # Check if it's actually a hardcoded key
                                        if any(sensitive in line.upper() for sensitive in self.SENSITIVE_VARS):
                                            if '=' in line and 'os.getenv' not in line and 'process.env' not in line:
                                                exposed.append((filepath, line_num, line.strip()))
                    except:
                        pass
        
        return exposed
    
    def validate_env_variables(self) -> Dict[str, str]:
        """Validate that all required environment variables are set"""
        status = {}
        
        # Check critical API keys
        critical_keys = [
            'SECRET_KEY',
            'JWT_SECRET_KEY',
            'DATABASE_URL'
        ]
        
        # Check market data APIs (at least one should be set)
        market_apis = [
            'POLYGON_API_KEY',
            'ALPHA_VANTAGE_API_KEY',
            'FINNHUB_API_KEY',
            'TWELVEDATA_API_KEY',
            'FMP_API_KEY'
        ]
        
        # Check AI APIs (optional but recommended)
        ai_apis = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY'
        ]
        
        # Validate critical keys
        for key in critical_keys:
            value = os.getenv(key)
            if not value or value.startswith('your-') or value == 'change-this':
                status[key] = '❌ Not configured or using default'
                self.issues.append(f"Critical key {key} is not properly configured")
            else:
                status[key] = '✅ Configured'
        
        # Check market APIs
        configured_market = [key for key in market_apis if os.getenv(key)]
        if not configured_market:
            self.warnings.append("⚠️ No market data APIs configured")
        else:
            print(f"✅ Market APIs configured: {', '.join(configured_market)}")
        
        # Check AI APIs
        configured_ai = [key for key in ai_apis if os.getenv(key)]
        if not configured_ai:
            self.warnings.append("⚠️ No AI/LLM APIs configured - AI features will be limited")
        else:
            print(f"✅ AI APIs configured: {', '.join(configured_ai)}")
        
        return status
    
    def check_git_history(self) -> bool:
        """Check if any secrets were previously committed"""
        import subprocess
        
        try:
            # Search git history for potential secrets
            result = subprocess.run(
                ['git', 'log', '--all', '--grep=API_KEY', '--grep=SECRET', '--grep=PASSWORD'],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                self.warnings.append("⚠️ Found potential secrets in git history. Consider using git-filter-repo to clean history.")
                return False
            
            return True
        except:
            self.warnings.append("⚠️ Could not check git history")
            return True
    
    def generate_secure_env_template(self):
        """Generate a secure .env.template file"""
        template = """# GoldenSignalsAI Security Configuration Template
# NEVER commit actual keys to GitHub!
# Copy this to .env and fill with your actual values

# ============= CRITICAL SECURITY =============
# Generate strong keys: python -c "import secrets; print(secrets.token_urlsafe(32))"
SECRET_KEY=CHANGE_THIS_USE_SCRIPT_ABOVE
JWT_SECRET_KEY=CHANGE_THIS_USE_SCRIPT_ABOVE

# ============= DATABASE =============
# For production, use PostgreSQL with SSL
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/dbname?ssl=require

# ============= API KEYS =============
# Get your keys from the providers' websites
# Store in environment variables or secret manager in production

# AI/LLM APIs (Optional but recommended)
OPENAI_API_KEY=sk-...  # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=sk-ant-...  # https://console.anthropic.com/

# Market Data APIs (Need at least one)
POLYGON_API_KEY=  # https://polygon.io/dashboard/api-keys
ALPHA_VANTAGE_API_KEY=  # https://www.alphavantage.co/support/#api-key
FINNHUB_API_KEY=  # https://finnhub.io/dashboard
TWELVEDATA_API_KEY=  # https://twelvedata.com/account/api-keys

# ============= PRODUCTION SECURITY =============
# Use these settings in production:
ENVIRONMENT=production
DEBUG=false
SECURE_SSL_REDIRECT=true
SESSION_COOKIE_SECURE=true
CSRF_COOKIE_SECURE=true
"""
        
        with open('.env.template', 'w') as f:
            f.write(template)
        
        print("✅ Generated secure .env.template file")
    
    def create_key_rotation_script(self):
        """Create a script for key rotation"""
        script = """#!/usr/bin/env python3
'''Key Rotation Helper - Rotate API keys safely'''

import secrets
import json
from datetime import datetime
from cryptography.fernet import Fernet

def generate_secure_key(length=32):
    '''Generate a cryptographically secure key'''
    return secrets.token_urlsafe(length)

def rotate_keys():
    '''Rotate all security keys'''
    
    # Generate new keys
    new_keys = {
        'SECRET_KEY': generate_secure_key(32),
        'JWT_SECRET_KEY': generate_secure_key(32),
        'ENCRYPTION_KEY': Fernet.generate_key().decode(),
        'ROTATION_DATE': datetime.now().isoformat()
    }
    
    # Save to secure location (NOT in repo)
    with open('.keys.json', 'w') as f:
        json.dump(new_keys, f, indent=2)
    
    print("🔐 New keys generated:")
    print(f"SECRET_KEY={new_keys['SECRET_KEY']}")
    print(f"JWT_SECRET_KEY={new_keys['JWT_SECRET_KEY']}")
    print(f"\\nAdd these to your .env file or secret manager")
    print("⚠️ Remember to update all running services!")

if __name__ == '__main__':
    rotate_keys()
"""
        
        with open('rotate_keys.py', 'w') as f:
            f.write(script)
        
        os.chmod('rotate_keys.py', 0o755)
        print("✅ Created key rotation script: rotate_keys.py")
    
    def setup_secrets_manager(self):
        """Setup instructions for cloud secret managers"""
        instructions = """
# 🔐 Production Secret Management Best Practices

## Option 1: AWS Secrets Manager
```python
import boto3
import json

def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
secrets = get_secret('goldensignals/production')
os.environ['OPENAI_API_KEY'] = secrets['OPENAI_API_KEY']
```

## Option 2: HashiCorp Vault
```python
import hvac

client = hvac.Client(url='https://vault.yourcompany.com')
client.token = os.environ['VAULT_TOKEN']
secret = client.secrets.kv.v2.read_secret_version(path='goldensignals')
os.environ.update(secret['data']['data'])
```

## Option 3: Azure Key Vault
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://your-vault.vault.azure.net/", credential=credential)
secret = client.get_secret("OPENAI-API-KEY")
os.environ['OPENAI_API_KEY'] = secret.value
```

## Option 4: Docker Secrets (for Docker deployments)
```yaml
# docker-compose.yml
version: '3.7'
services:
  app:
    image: goldensignals
    secrets:
      - openai_key
      - db_password
    environment:
      OPENAI_API_KEY_FILE: /run/secrets/openai_key

secrets:
  openai_key:
    external: true
  db_password:
    external: true
```

## Environment-Specific Configuration
```python
# config.py
class Config:
    def __init__(self):
        self.env = os.getenv('ENVIRONMENT', 'development')
        
        if self.env == 'production':
            # Load from secret manager
            self._load_from_secrets_manager()
        else:
            # Load from .env file
            from dotenv import load_dotenv
            load_dotenv()
    
    def _load_from_secrets_manager(self):
        # Your secret manager logic here
        pass
```
"""
        
        with open('SECRETS_MANAGEMENT.md', 'w') as f:
            f.write(instructions)
        
        print("✅ Created secrets management guide: SECRETS_MANAGEMENT.md")
    
    def run_full_check(self):
        """Run complete security audit"""
        print("\n" + "="*60)
        print("🔐 GOLDENSIGNALS SECURITY AUDIT")
        print("="*60 + "\n")
        
        # Check 1: .env file safety
        print("1️⃣ Checking .env file safety...")
        self.check_env_file_safety()
        
        # Check 2: Scan for exposed keys
        print("\n2️⃣ Scanning for hardcoded keys...")
        exposed = self.scan_for_exposed_keys()
        if exposed:
            print(f"❌ Found {len(exposed)} potential exposed keys:")
            for filepath, line_num, content in exposed[:5]:  # Show first 5
                print(f"  - {filepath}:{line_num}")
                print(f"    {content[:80]}...")
            self.issues.append(f"Found {len(exposed)} hardcoded keys in source code")
        else:
            print("✅ No hardcoded keys found in source code")
        
        # Check 3: Validate environment variables
        print("\n3️⃣ Validating environment variables...")
        env_status = self.validate_env_variables()
        
        # Check 4: Git history
        print("\n4️⃣ Checking git history...")
        self.check_git_history()
        
        # Generate security files
        print("\n5️⃣ Generating security helpers...")
        self.generate_secure_env_template()
        self.create_key_rotation_script()
        self.setup_secrets_manager()
        
        # Summary
        print("\n" + "="*60)
        print("📊 SECURITY AUDIT SUMMARY")
        print("="*60)
        
        if self.issues:
            print("\n❌ CRITICAL ISSUES (Fix immediately):")
            for issue in self.issues:
                print(f"  • {issue}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("  1. NEVER commit .env files to GitHub")
        print("  2. Use environment variables or secret managers in production")
        print("  3. Rotate keys regularly (use rotate_keys.py)")
        print("  4. Use different keys for dev/staging/production")
        print("  5. Enable 2FA on all API provider accounts")
        print("  6. Monitor API usage for unusual activity")
        print("  7. Use API key restrictions (IP, domain, etc.)")
        
        # GitHub Actions Secret Setup
        print("\n🔧 TO SETUP GITHUB ACTIONS SECRETS:")
        print("  1. Go to: Settings → Secrets and variables → Actions")
        print("  2. Add these secrets:")
        print("     - OPENAI_API_KEY")
        print("     - ANTHROPIC_API_KEY")
        print("     - DATABASE_URL (production)")
        print("     - Add other API keys as needed")
        print("\n  3. Access in workflows:")
        print("     env:")
        print("       OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}")
        
        print("\n" + "="*60)
        
        # Return status
        return len(self.issues) == 0

if __name__ == "__main__":
    checker = SecurityChecker()
    success = checker.run_full_check()
    
    if not success:
        print("\n⛔ SECURITY ISSUES DETECTED - Fix before deploying!")
        sys.exit(1)
    else:
        print("\n✅ Security check passed!")
        sys.exit(0)