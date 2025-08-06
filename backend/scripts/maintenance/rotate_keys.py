#!/usr/bin/env python3
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
    print(f"\nAdd these to your .env file or secret manager")
    print("⚠️ Remember to update all running services!")

if __name__ == '__main__':
    rotate_keys()
