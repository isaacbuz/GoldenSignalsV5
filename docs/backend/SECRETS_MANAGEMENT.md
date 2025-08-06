
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
import os

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
