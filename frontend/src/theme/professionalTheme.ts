import { createTheme } from '@mui/material/styles';

// Professional Bloomberg-inspired dark theme
export const professionalTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#ffb000',  // Bloomberg amber
      light: '#ffc947',
      dark: '#c68400',
    },
    secondary: {
      main: '#0068ff',  // Bright blue accent
      light: '#5a9fff',
      dark: '#0052cc',
    },
    background: {
      default: '#000000',  // Pure black like Bloomberg Terminal
      paper: '#0a0a0a',    // Slightly elevated surfaces
    },
    text: {
      primary: '#ffffff',
      secondary: '#b3b3b3',
    },
    success: {
      main: '#4af6c3',  // Bright cyan-green for positive movements
      light: '#7ffdd4',
      dark: '#00e5a0',
    },
    error: {
      main: '#ff433d',  // Bright red for negative movements
      light: '#ff6b66',
      dark: '#cc0000',
    },
    warning: {
      main: '#fb8b1e',  // Orange for warnings
      light: '#ffb347',
      dark: '#f57c00',
    },
    info: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    divider: 'rgba(255, 255, 255, 0.08)',
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    // Reduced font sizes for professional density
    h1: {
      fontWeight: 600,
      fontSize: '1.75rem',  // 28px
      letterSpacing: '-0.02em',
      lineHeight: 1.2,
    },
    h2: {
      fontWeight: 600,
      fontSize: '1.5rem',   // 24px
      letterSpacing: '-0.01em',
      lineHeight: 1.3,
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.25rem',  // 20px
      letterSpacing: '-0.01em',
      lineHeight: 1.4,
    },
    h4: {
      fontWeight: 500,
      fontSize: '1.125rem', // 18px
      lineHeight: 1.4,
    },
    h5: {
      fontWeight: 500,
      fontSize: '1rem',     // 16px
      lineHeight: 1.5,
    },
    h6: {
      fontWeight: 500,
      fontSize: '0.875rem', // 14px
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '0.875rem', // 14px - professional standard
      lineHeight: 1.5,
    },
    body2: {
      fontSize: '0.75rem',  // 12px - for dense data
      lineHeight: 1.4,
    },
    caption: {
      fontSize: '0.625rem', // 10px - for auxiliary information
      letterSpacing: '0.01em',
      lineHeight: 1.2,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
      fontSize: '0.875rem',
      letterSpacing: '0.02em',
    },
    overline: {
      fontSize: '0.625rem',
      fontWeight: 600,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      lineHeight: 1.2,
    },
  },
  shape: {
    borderRadius: 4, // Reduced from 8 for cleaner look
  },
  spacing: 8, // 8px base unit
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: '#333 #000',
          '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
            width: 8,
            height: 8,
          },
          '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
            borderRadius: 4,
            backgroundColor: '#333',
            border: '2px solid #000',
          },
          '&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track': {
            backgroundColor: '#000',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          padding: '6px 16px',
          transition: 'all 0.2s ease',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: '0 2px 8px rgba(255, 176, 0, 0.25)',
          },
        },
        sizeSmall: {
          padding: '4px 12px',
          fontSize: '0.75rem',
        },
        sizeLarge: {
          padding: '8px 20px',
          fontSize: '0.875rem',
        },
        containedPrimary: {
          background: '#ffb000',
          color: '#000000',
          fontWeight: 600,
          '&:hover': {
            background: '#ffc947',
          },
        },
        outlinedPrimary: {
          borderColor: '#ffb000',
          color: '#ffb000',
          '&:hover': {
            borderColor: '#ffc947',
            backgroundColor: 'rgba(255, 176, 0, 0.08)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#0a0a0a',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          transition: 'all 0.2s ease',
          '&:hover': {
            borderColor: 'rgba(255, 176, 0, 0.3)',
            transform: 'translateY(-1px)',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          height: 24,
          fontSize: '0.625rem',
          fontWeight: 600,
        },
        sizeSmall: {
          height: 20,
          fontSize: '0.625rem',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          fontSize: '0.75rem',
          padding: '8px 12px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        },
        head: {
          fontWeight: 600,
          fontSize: '0.625rem',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: '#b3b3b3',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiInputBase-input': {
            fontSize: '0.875rem',
          },
        },
      },
      defaultProps: {
        size: 'small',
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: '#1a1a1a',
          fontSize: '0.625rem',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#0a0a0a',
        },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: {
          borderColor: 'rgba(255, 255, 255, 0.08)',
        },
      },
    },
  },
});

// Alternative TradingView-inspired theme
export const tradingViewTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2962ff',
      light: '#768fff',
      dark: '#0039cb',
    },
    secondary: {
      main: '#ff9800',
      light: '#ffc947',
      dark: '#c66900',
    },
    background: {
      default: '#131722',
      paper: '#1e222d',
    },
    text: {
      primary: '#d1d4dc',
      secondary: '#787b86',
    },
    success: {
      main: '#26a69a',
      light: '#64d8cb',
      dark: '#00766c',
    },
    error: {
      main: '#ef5350',
      light: '#ff867c',
      dark: '#b61827',
    },
    warning: {
      main: '#ff9800',
      light: '#ffb74d',
      dark: '#f57c00',
    },
    info: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    divider: 'rgba(120, 123, 134, 0.2)',
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: 12,
    h1: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.125rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '0.875rem',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    h6: {
      fontSize: '0.75rem',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    body1: {
      fontSize: '0.8125rem',
      lineHeight: 1.5,
    },
    body2: {
      fontSize: '0.75rem',
      lineHeight: 1.4,
    },
    caption: {
      fontSize: '0.625rem',
      lineHeight: 1.2,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
      fontSize: '0.8125rem',
    },
  },
  shape: {
    borderRadius: 3,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 3,
          padding: '5px 12px',
          fontSize: '0.8125rem',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#1e222d',
          border: '1px solid #2a2e39',
        },
      },
    },
  },
});