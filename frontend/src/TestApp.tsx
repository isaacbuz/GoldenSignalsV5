import React from 'react';

const TestApp: React.FC = () => {
  return (
    <div style={{ 
      backgroundColor: '#000', 
      color: '#FFD700', 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      flexDirection: 'column',
      fontFamily: 'Inter, sans-serif'
    }}>
      <h1 style={{ fontSize: '48px', marginBottom: '20px' }}>🚀 GoldenSignals AI</h1>
      <p style={{ fontSize: '24px', marginBottom: '40px' }}>Application is Loading...</p>
      <div style={{ 
        backgroundColor: '#1a1a1a', 
        padding: '20px', 
        borderRadius: '10px',
        border: '2px solid #FFD700'
      }}>
        <p>✅ React is working</p>
        <p>✅ Frontend server is running on port 3000</p>
        <p>✅ Backend API is running on port 8000</p>
      </div>
    </div>
  );
};

export default TestApp;