import React, { useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Box,
  Chip,
  Skeleton,
  Tooltip,
  Button,
  useTheme,
} from '@mui/material';
import {
  Delete,
  TrendingUp,
  TrendingDown,
  Add,
  Refresh,
} from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import {
  setSelectedSymbol,
  removeFromWatchlist,
  fetchMultipleQuotes,
  addToWatchlist,
} from '../../store/slices/marketDataSlice';

const Watchlist: React.FC = () => {
  const theme = useTheme();
  const dispatch = useDispatch<AppDispatch>();
  const { watchlist, quotes, selectedSymbol, loading } = useSelector(
    (state: RootState) => state.marketData
  );

  useEffect(() => {
    // Fetch quotes for all watchlist items
    if (watchlist.length > 0) {
      dispatch(fetchMultipleQuotes(watchlist));
    }

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      if (watchlist.length > 0) {
        dispatch(fetchMultipleQuotes(watchlist));
      }
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [watchlist, dispatch]);

  const handleSelectSymbol = (symbol: string) => {
    dispatch(setSelectedSymbol(symbol));
  };

  const handleRemoveFromWatchlist = (
    event: React.MouseEvent,
    symbol: string
  ) => {
    event.stopPropagation();
    dispatch(removeFromWatchlist(symbol));
  };

  const handleRefresh = () => {
    if (watchlist.length > 0) {
      dispatch(fetchMultipleQuotes(watchlist));
    }
  };

  const formatNumber = (num: number, decimals: number = 2) => {
    return num?.toFixed(decimals) || '0.00';
  };

  const getPriceChangeChip = (change: number, changePercent: number) => {
    const isPositive = change >= 0;
    const icon = isPositive ? <TrendingUp /> : <TrendingDown />;
    const color = isPositive ? 'success' : 'error';
    const label = `${isPositive ? '+' : ''}${formatNumber(
      change
    )} (${formatNumber(changePercent)}%)`;

    return (
      <Chip
        icon={icon}
        label={label}
        color={color}
        size="small"
        sx={{ fontWeight: 'medium' }}
      />
    );
  };

  if (loading && Object.keys(quotes).length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Watchlist
          </Typography>
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} height={60} sx={{ my: 1 }} />
          ))}
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6" sx={{ fontSize: '0.875rem', fontWeight: 600 }}>
            Watchlist
          </Typography>
          <Box>
            <Tooltip title="Refresh quotes">
              <IconButton size="small" onClick={handleRefresh} disabled={loading}>
                <Refresh sx={{ fontSize: 16 }} />
              </IconButton>
            </Tooltip>
            <Tooltip title="Add to watchlist">
              <IconButton
                size="small"
                onClick={() => {
                  console.log('Add to watchlist');
                }}
              >
                <Add sx={{ fontSize: 16 }} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Box>

      <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 1 }}>
        {watchlist.length === 0 ? (
          <Box textAlign="center" py={4}>
            <Typography variant="caption" color="text.secondary" gutterBottom>
              No stocks in watchlist
            </Typography>
            <Button
              variant="outlined"
              size="small"
              startIcon={<Add />}
              sx={{ mt: 1, fontSize: '0.75rem' }}
            >
              Add Stocks
            </Button>
          </Box>
        ) : (
          <List sx={{ mx: -1 }}>
            {watchlist.map((symbol) => {
              const quote = quotes[symbol];
              const isSelected = selectedSymbol === symbol;

              return (
                <ListItem key={symbol} disablePadding>
                  <ListItemButton
                    onClick={() => handleSelectSymbol(symbol)}
                    selected={isSelected}
                    sx={{
                      borderRadius: 1,
                      mb: 0.5,
                      '&.Mui-selected': {
                        backgroundColor: 'action.selected',
                      },
                    }}
                  >
                    <ListItemText
                      primary={
                        <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                          {symbol}
                        </Typography>
                      }
                      secondary={
                        quote ? (
                          <Box>
                            <Typography variant="body2" sx={{ fontWeight: 500, fontSize: '0.875rem' }}>
                              ${formatNumber(quote.price)}
                            </Typography>
                            <Typography 
                              variant="caption" 
                              sx={{ 
                                color: quote.change >= 0 ? theme.palette.success.main : theme.palette.error.main,
                                fontWeight: 500,
                                fontSize: '0.625rem',
                              }}
                            >
                              {quote.change >= 0 ? '+' : ''}{formatNumber(quote.change)} ({formatNumber(quote.changePercent)}%)
                            </Typography>
                          </Box>
                        ) : (
                          <Typography variant="caption" color="text.secondary">
                            Loading...
                          </Typography>
                        )
                      }
                    />
                    <ListItemSecondaryAction>
                      <Tooltip title="Remove from watchlist">
                        <IconButton
                          edge="end"
                          size="small"
                          onClick={(e) => handleRemoveFromWatchlist(e, symbol)}
                        >
                          <Delete />
                        </IconButton>
                      </Tooltip>
                    </ListItemSecondaryAction>
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        )}
      </Box>
    </Box>
  );
};

export default Watchlist;