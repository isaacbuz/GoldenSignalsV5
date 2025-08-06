"""
Authentication API Tests
Comprehensive testing of user authentication and authorization endpoints
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import status
import jwt

from app import app
from core.config import settings


client = TestClient(app)


class TestAuthAPI:
    """Test suite for authentication API endpoints"""
    
    def test_register_success(self):
        """Test successful user registration"""
        with patch('api.routes.auth.get_session') as mock_session, \
             patch('api.routes.auth.User') as mock_user_model, \
             patch('api.routes.auth.get_password_hash') as mock_hash:
            
            # Mock database session
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            # Mock user creation
            mock_user = Mock()
            mock_user.id = 1
            mock_user.email = "test@example.com"
            mock_user.username = "testuser"
            mock_user_model.return_value = mock_user
            
            # Mock password hashing
            mock_hash.return_value = "hashed_password"
            
            # Mock database operations
            mock_session_instance.add = Mock()
            mock_session_instance.commit = AsyncMock()
            mock_session_instance.refresh = AsyncMock()
            
            # Mock existing user check
            mock_query = Mock()
            mock_query.filter.return_value.first = AsyncMock(return_value=None)
            mock_session_instance.query.return_value = mock_query
            
            user_data = {
                "username": "testuser",
                "email": "test@example.com",
                "password": "Test123!",
                "full_name": "Test User"
            }
            
            response = client.post("/api/v1/auth/register", json=user_data)
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["message"] == "User registered successfully"
            assert data["user"]["email"] == "test@example.com"
    
    def test_register_duplicate_email(self):
        """Test registration with duplicate email"""
        with patch('api.routes.auth.get_session') as mock_session:
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            # Mock existing user found
            existing_user = Mock()
            existing_user.email = "test@example.com"
            mock_query = Mock()
            mock_query.filter.return_value.first = AsyncMock(return_value=existing_user)
            mock_session_instance.query.return_value = mock_query
            
            user_data = {
                "username": "testuser",
                "email": "test@example.com",
                "password": "Test123!",
                "full_name": "Test User"
            }
            
            response = client.post("/api/v1/auth/register", json=user_data)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            data = response.json()
            assert "already registered" in data["detail"].lower()
    
    def test_register_invalid_password(self):
        """Test registration with invalid password"""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "weak",  # Too weak
            "full_name": "Test User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_register_invalid_email(self):
        """Test registration with invalid email format"""
        user_data = {
            "username": "testuser",
            "email": "invalid-email",
            "password": "Test123!",
            "full_name": "Test User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_login_success(self):
        """Test successful user login"""
        with patch('api.routes.auth.authenticate_user') as mock_auth, \
             patch('api.routes.auth.create_access_token') as mock_token:
            
            # Mock user authentication
            mock_user = Mock()
            mock_user.id = 1
            mock_user.email = "test@example.com"
            mock_user.username = "testuser"
            mock_user.is_active = True
            mock_auth.return_value = mock_user
            
            # Mock token creation
            mock_token.return_value = "test_access_token"
            
            login_data = {
                "username": "test@example.com",
                "password": "Test123!"
            }
            
            response = client.post("/api/v1/auth/login", data=login_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["access_token"] == "test_access_token"
            assert data["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        with patch('api.routes.auth.authenticate_user') as mock_auth:
            mock_auth.return_value = None  # Authentication failed
            
            login_data = {
                "username": "test@example.com",
                "password": "wrongpassword"
            }
            
            response = client.post("/api/v1/auth/login", data=login_data)
            
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            data = response.json()
            assert "incorrect" in data["detail"].lower()
    
    def test_login_inactive_user(self):
        """Test login with inactive user account"""
        with patch('api.routes.auth.authenticate_user') as mock_auth:
            mock_user = Mock()
            mock_user.is_active = False
            mock_auth.return_value = mock_user
            
            login_data = {
                "username": "test@example.com",
                "password": "Test123!"
            }
            
            response = client.post("/api/v1/auth/login", data=login_data)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            data = response.json()
            assert "inactive" in data["detail"].lower()
    
    def test_get_current_user_success(self):
        """Test successful current user retrieval"""
        with patch('api.routes.auth.get_current_user') as mock_get_user:
            mock_user = Mock()
            mock_user.id = 1
            mock_user.email = "test@example.com"
            mock_user.username = "testuser"
            mock_user.full_name = "Test User"
            mock_user.is_active = True
            mock_get_user.return_value = mock_user
            
            # Create a valid token
            token_data = {"sub": "test@example.com", "exp": datetime.utcnow() + timedelta(hours=1)}
            token = jwt.encode(token_data, settings.SECRET_KEY, algorithm="HS256")
            
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/api/v1/auth/me", headers=headers)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["email"] == "test@example.com"
            assert data["username"] == "testuser"
    
    def test_get_current_user_no_token(self):
        """Test current user retrieval without token"""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user_invalid_token(self):
        """Test current user retrieval with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user_expired_token(self):
        """Test current user retrieval with expired token"""
        # Create expired token
        token_data = {"sub": "test@example.com", "exp": datetime.utcnow() - timedelta(hours=1)}
        token = jwt.encode(token_data, settings.SECRET_KEY, algorithm="HS256")
        
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_refresh_token_success(self):
        """Test successful token refresh"""
        with patch('api.routes.auth.get_current_user') as mock_get_user, \
             patch('api.routes.auth.create_access_token') as mock_token:
            
            mock_user = Mock()
            mock_user.email = "test@example.com"
            mock_user.is_active = True
            mock_get_user.return_value = mock_user
            
            mock_token.return_value = "new_access_token"
            
            # Valid token for refresh
            token_data = {"sub": "test@example.com", "exp": datetime.utcnow() + timedelta(hours=1)}
            token = jwt.encode(token_data, settings.SECRET_KEY, algorithm="HS256")
            
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post("/api/v1/auth/refresh", headers=headers)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["access_token"] == "new_access_token"
    
    def test_change_password_success(self):
        """Test successful password change"""
        with patch('api.routes.auth.get_current_user') as mock_get_user, \
             patch('api.routes.auth.get_session') as mock_session, \
             patch('api.routes.auth.verify_password') as mock_verify, \
             patch('api.routes.auth.get_password_hash') as mock_hash:
            
            mock_user = Mock()
            mock_user.id = 1
            mock_user.email = "test@example.com"
            mock_user.hashed_password = "old_hashed_password"
            mock_get_user.return_value = mock_user
            
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.commit = AsyncMock()
            
            mock_verify.return_value = True
            mock_hash.return_value = "new_hashed_password"
            
            token_data = {"sub": "test@example.com", "exp": datetime.utcnow() + timedelta(hours=1)}
            token = jwt.encode(token_data, settings.SECRET_KEY, algorithm="HS256")
            
            change_data = {
                "current_password": "OldPass123!",
                "new_password": "NewPass123!"
            }
            
            headers = {"Authorization": f"Bearer {token}"}
            response = client.put("/api/v1/auth/change-password", 
                                json=change_data, headers=headers)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["message"] == "Password updated successfully"
    
    def test_change_password_wrong_current(self):
        """Test password change with wrong current password"""
        with patch('api.routes.auth.get_current_user') as mock_get_user, \
             patch('api.routes.auth.verify_password') as mock_verify:
            
            mock_user = Mock()
            mock_user.hashed_password = "old_hashed_password"
            mock_get_user.return_value = mock_user
            
            mock_verify.return_value = False  # Wrong current password
            
            token_data = {"sub": "test@example.com", "exp": datetime.utcnow() + timedelta(hours=1)}
            token = jwt.encode(token_data, settings.SECRET_KEY, algorithm="HS256")
            
            change_data = {
                "current_password": "WrongPass123!",
                "new_password": "NewPass123!"
            }
            
            headers = {"Authorization": f"Bearer {token}"}
            response = client.put("/api/v1/auth/change-password",
                                json=change_data, headers=headers)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_forgot_password_success(self):
        """Test successful forgot password request"""
        with patch('api.routes.auth.get_session') as mock_session, \
             patch('api.routes.auth.User') as mock_user_model, \
             patch('api.routes.auth.send_reset_email') as mock_send_email:
            
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            mock_user = Mock()
            mock_user.email = "test@example.com"
            mock_query = Mock()
            mock_query.filter.return_value.first = AsyncMock(return_value=mock_user)
            mock_session_instance.query.return_value = mock_query
            
            mock_send_email.return_value = True
            
            response = client.post("/api/v1/auth/forgot-password",
                                 json={"email": "test@example.com"})
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "reset link" in data["message"].lower()
    
    def test_forgot_password_user_not_found(self):
        """Test forgot password for non-existent user"""
        with patch('api.routes.auth.get_session') as mock_session:
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            mock_query = Mock()
            mock_query.filter.return_value.first = AsyncMock(return_value=None)
            mock_session_instance.query.return_value = mock_query
            
            response = client.post("/api/v1/auth/forgot-password",
                                 json={"email": "nonexistent@example.com"})
            
            # Should return 200 to avoid user enumeration
            assert response.status_code == status.HTTP_200_OK
    
    def test_reset_password_success(self):
        """Test successful password reset"""
        with patch('api.routes.auth.verify_reset_token') as mock_verify_token, \
             patch('api.routes.auth.get_session') as mock_session, \
             patch('api.routes.auth.get_password_hash') as mock_hash:
            
            mock_verify_token.return_value = "test@example.com"
            
            mock_session_instance = Mock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.commit = AsyncMock()
            
            mock_user = Mock()
            mock_query = Mock()
            mock_query.filter.return_value.first = AsyncMock(return_value=mock_user)
            mock_session_instance.query.return_value = mock_query
            
            mock_hash.return_value = "new_hashed_password"
            
            reset_data = {
                "token": "valid_reset_token",
                "new_password": "NewPass123!"
            }
            
            response = client.post("/api/v1/auth/reset-password", json=reset_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["message"] == "Password reset successfully"
    
    def test_reset_password_invalid_token(self):
        """Test password reset with invalid token"""
        with patch('api.routes.auth.verify_reset_token') as mock_verify_token:
            mock_verify_token.return_value = None  # Invalid token
            
            reset_data = {
                "token": "invalid_reset_token",
                "new_password": "NewPass123!"
            }
            
            response = client.post("/api/v1/auth/reset-password", json=reset_data)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_logout_success(self):
        """Test successful logout"""
        with patch('api.routes.auth.get_current_user') as mock_get_user, \
             patch('api.routes.auth.blacklist_token') as mock_blacklist:
            
            mock_user = Mock()
            mock_user.email = "test@example.com"
            mock_get_user.return_value = mock_user
            
            mock_blacklist.return_value = True
            
            token_data = {"sub": "test@example.com", "exp": datetime.utcnow() + timedelta(hours=1)}
            token = jwt.encode(token_data, settings.SECRET_KEY, algorithm="HS256")
            
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post("/api/v1/auth/logout", headers=headers)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["message"] == "Logged out successfully"


@pytest.mark.integration
class TestAuthAPIIntegration:
    """Integration tests for authentication"""
    
    def test_full_auth_flow(self):
        """Test complete authentication flow"""
        pytest.skip("Requires database setup - run manually")
    
    def test_token_security(self):
        """Test token security measures"""
        pytest.skip("Security test - run manually")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])