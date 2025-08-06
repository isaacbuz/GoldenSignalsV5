"""
Tests for authentication endpoints and security
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.database import Base, get_db
from app import app
from core.security import get_password_hash

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_register_user(self):
        """Test user registration"""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "testuser",
                "password": "SecurePassword123!",
                "full_name": "Test User"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert "id" in data
        assert "hashed_password" not in data  # Should not expose password
    
    def test_register_duplicate_user(self):
        """Test registering duplicate user"""
        # First registration
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@example.com",
                "username": "duplicate",
                "password": "Password123!",
                "full_name": "Duplicate User"
            }
        )
        
        # Attempt duplicate registration
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@example.com",
                "username": "duplicate",
                "password": "Password123!",
                "full_name": "Duplicate User"
            }
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    
    def test_login(self):
        """Test user login"""
        # Register user first
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "login@example.com",
                "username": "loginuser",
                "password": "LoginPassword123!",
                "full_name": "Login User"
            }
        )
        
        # Login
        response = client.post(
            "/api/v1/auth/token",
            data={
                "username": "loginuser",
                "password": "LoginPassword123!"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = client.post(
            "/api/v1/auth/token",
            data={
                "username": "nonexistent",
                "password": "WrongPassword"
            }
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_get_current_user(self):
        """Test getting current user info"""
        # Register and login
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "current@example.com",
                "username": "currentuser",
                "password": "CurrentPassword123!",
                "full_name": "Current User"
            }
        )
        
        login_response = client.post(
            "/api/v1/auth/token",
            data={
                "username": "currentuser",
                "password": "CurrentPassword123!"
            }
        )
        token = login_response.json()["access_token"]
        
        # Get current user
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "currentuser"
        assert data["email"] == "current@example.com"
    
    def test_refresh_token(self):
        """Test token refresh"""
        # Register and login
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "refresh@example.com",
                "username": "refreshuser",
                "password": "RefreshPassword123!",
                "full_name": "Refresh User"
            }
        )
        
        login_response = client.post(
            "/api/v1/auth/token",
            data={
                "username": "refreshuser",
                "password": "RefreshPassword123!"
            }
        )
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh token
        response = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    def test_change_password(self):
        """Test password change"""
        # Register and login
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "changepass@example.com",
                "username": "changepassuser",
                "password": "OldPassword123!",
                "full_name": "Change Pass User"
            }
        )
        
        login_response = client.post(
            "/api/v1/auth/token",
            data={
                "username": "changepassuser",
                "password": "OldPassword123!"
            }
        )
        token = login_response.json()["access_token"]
        
        # Change password
        response = client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "OldPassword123!",
                "new_password": "NewPassword123!"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        # Try login with new password
        login_response = client.post(
            "/api/v1/auth/token",
            data={
                "username": "changepassuser",
                "password": "NewPassword123!"
            }
        )
        assert login_response.status_code == 200
    
    def test_protected_endpoint_without_token(self):
        """Test accessing protected endpoint without token"""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
    
    def test_protected_endpoint_with_invalid_token(self):
        """Test accessing protected endpoint with invalid token"""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]