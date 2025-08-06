import React, { useState, useEffect } from 'react';
import { Box, Container, Typography, Paper, Grid, Button, Alert } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#FFD700',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
});

function SimpleApp() {
  const [backendStatus, setBackendStatus] = useState<string>('Checking...');
  const [marketData, setMarketData] = useState<any>(null);

  useEffect(() => {
    // Check backend health
    fetch('http://localhost:8000/api/v1/health')
      .then(res => res.json())
      .then(data => setBackendStatus(data.status || 'Connected'))
      .catch(() => setBackendStatus('Not connected'));

    // Try to fetch some market data
    fetch('http://localhost:8000/api/v1/market/quote/AAPL')
      .then(res => res.json())
      .then(data => setMarketData(data))
      .catch(err => console.log('Market data error:', err));
  }, []);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Typography variant="h2" sx={{ color: '#FFD700', mb: 4, textAlign: 'center' }}>
          🚀 GoldenSignals AI Trading Platform
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h5" sx={{ mb: 2, color: '#FFD700' }}>
                System Status
              </Typography>
              <Alert severity={backendStatus === 'healthy' ? 'success' : 'info'} sx={{ mb: 2 }}>
                Backend API: {backendStatus}
              </Alert>
              <Typography>Frontend: ✅ Running on port 3000</Typography>
              <Typography>Backend: {backendStatus === 'healthy' ? '✅' : '⏳'} Port 8000</Typography>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h5" sx={{ mb: 2, color: '#FFD700' }}>
                Quick Actions
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Button 
                  variant="contained" 
                  href="/ai-dashboard"
                  sx={{ backgroundColor: '#FFD700', color: '#000' }}
                >
                  Open AI Dashboard
                </Button>
                <Button 
                  variant="outlined" 
                  href="http://localhost:8000/docs"
                  target="_blank"
                  sx={{ borderColor: '#FFD700', color: '#FFD700' }}
                >
                  View API Documentation
                </Button>
              </Box>
            </Paper>
          </Grid>

          {marketData && (
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" sx={{ mb: 2, color: '#FFD700' }}>
                  Market Data (AAPL)
                </Typography>
                <pre>{JSON.stringify(marketData, null, 2)}</pre>
              </Paper>
            </Grid>
          )}
        </Grid>

        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <Typography variant="body2" sx={{ color: '#666' }}>
            AI-Powered Trading with Intelligent Chart Analysis
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default SimpleApp;