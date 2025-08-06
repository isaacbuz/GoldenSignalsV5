import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import GoldenSignalsAIDashboard from './pages/GoldenSignalsAIDashboard';

const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/signals" replace />} />
      <Route path="/signals" element={<GoldenSignalsAIDashboard />} />
      <Route path="/flow" element={<GoldenSignalsAIDashboard />} />
      <Route path="/charts" element={<GoldenSignalsAIDashboard />} />
      <Route path="/positions" element={<GoldenSignalsAIDashboard />} />
      <Route path="/dashboard" element={<GoldenSignalsAIDashboard />} />
      <Route path="/settings" element={<GoldenSignalsAIDashboard />} />
    </Routes>
  );
};

export default AppRoutes;