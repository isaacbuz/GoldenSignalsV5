import React, { useState, useCallback, useEffect } from 'react';
import {
  Autocomplete,
  TextField,
  Box,
  Typography,
  Chip,
  CircularProgress,
  Paper,
  IconButton,
  InputAdornment
} from '@mui/material';
import { Search, TrendingUp } from '@mui/icons-material';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '../../store/store';
import { setSelectedSymbol } from '../../store/slices/marketDataSlice';
import axios from 'axios';
import debounce from 'lodash/debounce';

interface StockOption {
  symbol: string;
  name: string;
  type?: string;
  exchange?: string;
}

const POPULAR_STOCKS: StockOption[] = [
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.' },
  { symbol: 'MSFT', name: 'Microsoft Corporation' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.' },
  { symbol: 'TSLA', name: 'Tesla Inc.' },
  { symbol: 'META', name: 'Meta Platforms Inc.' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation' },
  { symbol: 'SPY', name: 'SPDR S&P 500 ETF' },
];

const StockSearch: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const [options, setOptions] = useState<StockOption[]>(POPULAR_STOCKS);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  // Debounced search function
  const searchStocks = useCallback(
    debounce(async (query: string) => {
      if (!query || query.length < 1) {
        setOptions(POPULAR_STOCKS);
        return;
      }

      setLoading(true);
      try {
        const response = await axios.get(`/api/v1/market-data/search`, {
          params: { query, limit: 10 }
        });
        
        const searchResults = response.data || [];
        
        // Combine search results with popular stocks for better UX
        const combinedResults = [
          ...searchResults,
          ...POPULAR_STOCKS.filter(stock => 
            stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
            stock.name.toLowerCase().includes(query.toLowerCase())
          )
        ];
        
        // Remove duplicates
        const uniqueResults = Array.from(
          new Map(combinedResults.map(item => [item.symbol, item])).values()
        );
        
        setOptions(uniqueResults.slice(0, 10));
      } catch (error) {
        console.error('Error searching stocks:', error);
        // Fallback to local search
        const filtered = POPULAR_STOCKS.filter(stock =>
          stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
          stock.name.toLowerCase().includes(query.toLowerCase())
        );
        setOptions(filtered);
      }
      setLoading(false);
    }, 300),
    []
  );

  useEffect(() => {
    if (inputValue) {
      searchStocks(inputValue);
    } else {
      setOptions(POPULAR_STOCKS);
    }
  }, [inputValue, searchStocks]);

  const handleSymbolSelect = (event: any, value: StockOption | null) => {
    if (value) {
      dispatch(setSelectedSymbol(value.symbol));
      setInputValue('');
    }
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 400 }}>
      <Autocomplete
        options={options}
        getOptionLabel={(option) => option.symbol}
        onOpen={() => setOpen(true)}
        onClose={() => setOpen(false)}
        open={open}
        loading={loading}
        inputValue={inputValue}
        onInputChange={(event, newInputValue) => {
          setInputValue(newInputValue);
        }}
        onChange={handleSymbolSelect}
        renderInput={(params) => (
          <TextField
            {...params}
            placeholder="Search stocks..."
            variant="outlined"
            size="small"
            InputProps={{
              ...params.InputProps,
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
              endAdornment: (
                <>
                  {loading ? <CircularProgress color="inherit" size={20} /> : null}
                  {params.InputProps.endAdornment}
                </>
              ),
              sx: {
                bgcolor: 'background.paper',
                '&:hover': {
                  bgcolor: 'action.hover',
                },
              }
            }}
          />
        )}
        renderOption={(props, option) => (
          <Box component="li" {...props}>
            <Box sx={{ width: '100%' }}>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography variant="body1" fontWeight="medium">
                    {option.symbol}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {option.name}
                  </Typography>
                </Box>
                {option.type && (
                  <Chip 
                    label={option.type} 
                    size="small" 
                    variant="outlined"
                    sx={{ ml: 1 }}
                  />
                )}
              </Box>
            </Box>
          </Box>
        )}
        PaperComponent={(props) => (
          <Paper {...props} elevation={8}>
            {!loading && inputValue === '' && (
              <Box sx={{ p: 1, borderBottom: '1px solid', borderColor: 'divider' }}>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                  <TrendingUp sx={{ fontSize: 16, mr: 0.5 }} />
                  Popular Stocks
                </Typography>
              </Box>
            )}
            {props.children}
          </Paper>
        )}
        sx={{
          '& .MuiAutocomplete-popupIndicator': {
            transform: 'none',
          },
        }}
      />
    </Box>
  );
};

export default StockSearch;