"""
Historical Data Storage System
Efficient storage and retrieval of market data, signals, and analytics
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
import json
import gzip
import pickle
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.dialects.postgresql import insert
import aiofiles
import pyarrow as pa
import pyarrow.parquet as pq

from database.models import MarketData, SignalHistory, AgentPerformance, BacktestResult
from database.connection import get_db
from core.logging import get_logger
from agents.base import Signal

logger = get_logger(__name__)


class DataFormat(Enum):
    """Data storage formats"""
    DATABASE = "database"  # PostgreSQL/SQLite
    PARQUET = "parquet"  # Apache Parquet files
    CSV = "csv"  # CSV files
    JSON = "json"  # JSON files
    PICKLE = "pickle"  # Python pickle
    HDF5 = "hdf5"  # HDF5 format


class TimeFrame(Enum):
    """Market data timeframes"""
    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class DataType(Enum):
    """Types of historical data"""
    OHLCV = "ohlcv"  # Price data
    TICK = "tick"  # Tick data
    ORDER_BOOK = "order_book"  # Order book snapshots
    TRADES = "trades"  # Trade executions
    SIGNALS = "signals"  # Trading signals
    METRICS = "metrics"  # Performance metrics
    INDICATORS = "indicators"  # Technical indicators
    NEWS = "news"  # News data
    SENTIMENT = "sentiment"  # Sentiment data


@dataclass
class StorageConfig:
    """Storage configuration"""
    # Database settings
    use_database: bool = True
    database_batch_size: int = 1000
    
    # File storage settings
    use_files: bool = True
    file_format: DataFormat = DataFormat.PARQUET
    base_path: str = "./data/historical"
    
    # Compression
    compress_files: bool = True
    compression_level: int = 6
    
    # Partitioning
    partition_by_date: bool = True
    partition_by_symbol: bool = True
    
    # Retention
    retention_days: Dict[DataType, int] = field(default_factory=lambda: {
        DataType.TICK: 7,
        DataType.OHLCV: 365,
        DataType.SIGNALS: 180,
        DataType.METRICS: 365
    })
    
    # Cache
    use_cache: bool = True
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 300


class DataQuery(BaseModel):
    """Query parameters for historical data"""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    data_type: DataType = DataType.OHLCV
    timeframe: Optional[TimeFrame] = TimeFrame.DAY_1
    
    # Filters
    min_volume: Optional[float] = None
    min_price: Optional[float] = None
    
    # Options
    include_indicators: bool = False
    include_signals: bool = False
    resample: Optional[str] = None  # Resample to different timeframe
    
    # Pagination
    limit: Optional[int] = None
    offset: int = 0


class DataPoint(BaseModel):
    """Single data point"""
    symbol: str
    timestamp: datetime
    data_type: DataType
    
    # OHLCV data
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    
    # Additional fields
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    
    # Signal data
    signal: Optional[Signal] = None
    agent_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HistoricalDataStorage:
    """
    Manages historical data storage and retrieval
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Cache
        self.cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.cache_size = 0
        
        # Batch buffers
        self.write_buffers: Dict[str, List[DataPoint]] = {}
        self.buffer_lock = asyncio.Lock()
        
        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._active = False
        
        logger.info(f"Historical data storage initialized at {self.base_path}")
    
    async def start(self) -> None:
        """Start background tasks"""
        if self._active:
            return
        
        self._active = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Historical data storage started")
    
    async def stop(self) -> None:
        """Stop background tasks"""
        self._active = False
        
        # Flush remaining buffers
        await self.flush_all()
        
        # Cancel tasks
        if self._flush_task:
            self._flush_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Historical data storage stopped")
    
    async def store_ohlcv(
        self,
        symbol: str,
        timeframe: TimeFrame,
        data: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> bool:
        """
        Store OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            data: OHLCV data (DataFrame or list of dicts)
            
        Returns:
            Success status
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Ensure required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in OHLCV data")
                return False
            
            # Add metadata
            df['symbol'] = symbol
            df['timeframe'] = timeframe.value
            
            # Store in database
            if self.config.use_database:
                await self._store_to_database(df, DataType.OHLCV)
            
            # Store in files
            if self.config.use_files:
                await self._store_to_file(df, symbol, DataType.OHLCV, timeframe)
            
            # Update cache
            if self.config.use_cache:
                self._update_cache(f"{symbol}_{timeframe.value}", df)
            
            logger.debug(f"Stored {len(df)} OHLCV records for {symbol} ({timeframe.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store OHLCV data: {str(e)}")
            return False
    
    async def store_signal(
        self,
        signal: Signal,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store trading signal
        
        Args:
            signal: Trading signal
            agent_id: Agent that generated signal
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            async with get_db() as session:
                # Create signal record
                signal_record = SignalHistory(
                    signal_id=f"{agent_id}_{datetime.now().timestamp()}",
                    agent_id=agent_id,
                    symbol=signal.symbol,
                    timestamp=datetime.now(),
                    signal_type=signal.action,
                    confidence=signal.confidence,
                    expected_return=signal.metadata.get("expected_return"),
                    stop_loss=signal.metadata.get("stop_loss"),
                    take_profit=signal.metadata.get("take_profit"),
                    market_conditions=metadata or {},
                    reasoning=signal.metadata.get("reasoning", "")
                )
                
                session.add(signal_record)
                await session.commit()
            
            # Also store in files for faster backtesting
            if self.config.use_files:
                signal_data = {
                    "timestamp": datetime.now(),
                    "agent_id": agent_id,
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "metadata": signal.metadata
                }
                
                await self._append_to_signal_file(signal.symbol, signal_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store signal: {str(e)}")
            return False
    
    async def store_metrics(
        self,
        metrics: Dict[str, Any],
        metric_type: str = "performance"
    ) -> bool:
        """
        Store performance metrics
        
        Args:
            metrics: Metrics data
            metric_type: Type of metrics
            
        Returns:
            Success status
        """
        try:
            # Store in database
            if self.config.use_database and "agent_id" in metrics:
                async with get_db() as session:
                    performance_record = AgentPerformance(
                        agent_id=metrics["agent_id"],
                        timestamp=datetime.now(),
                        accuracy=metrics.get("accuracy"),
                        win_rate=metrics.get("win_rate"),
                        sharpe_ratio=metrics.get("sharpe_ratio"),
                        total_pnl=metrics.get("total_pnl"),
                        max_drawdown=metrics.get("max_drawdown"),
                        metrics_json=metrics
                    )
                    
                    session.add(performance_record)
                    await session.commit()
            
            # Store in files
            if self.config.use_files:
                await self._append_to_metrics_file(metric_type, metrics)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {str(e)}")
            return False
    
    async def query(
        self,
        query: DataQuery
    ) -> pd.DataFrame:
        """
        Query historical data
        
        Args:
            query: Query parameters
            
        Returns:
            DataFrame with results
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            if self.config.use_cache and cache_key in self.cache:
                df, cached_time = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.config.cache_ttl_seconds:
                    logger.debug(f"Returning cached data for {cache_key}")
                    return df.copy()
            
            # Query based on data type
            if query.data_type == DataType.OHLCV:
                df = await self._query_ohlcv(query)
            elif query.data_type == DataType.SIGNALS:
                df = await self._query_signals(query)
            elif query.data_type == DataType.METRICS:
                df = await self._query_metrics(query)
            else:
                logger.warning(f"Unsupported data type: {query.data_type}")
                return pd.DataFrame()
            
            # Apply filters
            if query.min_volume and 'volume' in df.columns:
                df = df[df['volume'] >= query.min_volume]
            
            if query.min_price and 'close' in df.columns:
                df = df[df['close'] >= query.min_price]
            
            # Add indicators if requested
            if query.include_indicators and query.data_type == DataType.OHLCV:
                df = self._add_indicators(df)
            
            # Add signals if requested
            if query.include_signals:
                signals_df = await self._query_signals(query)
                if not signals_df.empty:
                    df = self._merge_signals(df, signals_df)
            
            # Resample if requested
            if query.resample:
                df = self._resample_data(df, query.resample)
            
            # Apply pagination
            if query.limit:
                df = df.iloc[query.offset:query.offset + query.limit]
            
            # Update cache
            if self.config.use_cache:
                self._update_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return pd.DataFrame()
    
    async def get_latest(
        self,
        symbol: str,
        data_type: DataType = DataType.OHLCV,
        timeframe: Optional[TimeFrame] = TimeFrame.DAY_1
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest data point
        
        Args:
            symbol: Trading symbol
            data_type: Type of data
            timeframe: Timeframe for OHLCV
            
        Returns:
            Latest data point
        """
        try:
            if data_type == DataType.OHLCV:
                async with get_db() as session:
                    result = await session.execute(
                        select(MarketData)
                        .where(
                            and_(
                                MarketData.symbol == symbol,
                                MarketData.timeframe == timeframe.value
                            )
                        )
                        .order_by(MarketData.timestamp.desc())
                        .limit(1)
                    )
                    
                    record = result.scalar_one_or_none()
                    if record:
                        return {
                            "symbol": record.symbol,
                            "timestamp": record.timestamp,
                            "open": record.open,
                            "high": record.high,
                            "low": record.low,
                            "close": record.close,
                            "volume": record.volume
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest data: {str(e)}")
            return None
    
    async def get_data_range(
        self,
        symbol: str,
        data_type: DataType = DataType.OHLCV
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get available data range for symbol
        
        Args:
            symbol: Trading symbol
            data_type: Type of data
            
        Returns:
            Tuple of (earliest, latest) timestamps
        """
        try:
            async with get_db() as session:
                if data_type == DataType.OHLCV:
                    result = await session.execute(
                        select(
                            func.min(MarketData.timestamp),
                            func.max(MarketData.timestamp)
                        ).where(MarketData.symbol == symbol)
                    )
                    
                    min_date, max_date = result.one()
                    if min_date and max_date:
                        return (min_date, max_date)
                
                elif data_type == DataType.SIGNALS:
                    result = await session.execute(
                        select(
                            func.min(SignalHistory.timestamp),
                            func.max(SignalHistory.timestamp)
                        ).where(SignalHistory.symbol == symbol)
                    )
                    
                    min_date, max_date = result.one()
                    if min_date and max_date:
                        return (min_date, max_date)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get data range: {str(e)}")
            return None
    
    async def export_data(
        self,
        query: DataQuery,
        output_path: str,
        format: DataFormat = DataFormat.CSV
    ) -> bool:
        """
        Export historical data to file
        
        Args:
            query: Query parameters
            output_path: Output file path
            format: Export format
            
        Returns:
            Success status
        """
        try:
            # Query data
            df = await self.query(query)
            
            if df.empty:
                logger.warning("No data to export")
                return False
            
            # Export based on format
            if format == DataFormat.CSV:
                df.to_csv(output_path, index=False)
            elif format == DataFormat.PARQUET:
                df.to_parquet(output_path, compression='snappy')
            elif format == DataFormat.JSON:
                df.to_json(output_path, orient='records', date_format='iso')
            elif format == DataFormat.PICKLE:
                df.to_pickle(output_path)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported {len(df)} records to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
    
    async def import_data(
        self,
        file_path: str,
        symbol: str,
        data_type: DataType,
        timeframe: Optional[TimeFrame] = None,
        format: Optional[DataFormat] = None
    ) -> bool:
        """
        Import historical data from file
        
        Args:
            file_path: Input file path
            symbol: Trading symbol
            data_type: Type of data
            timeframe: Timeframe for OHLCV
            format: File format (auto-detect if None)
            
        Returns:
            Success status
        """
        try:
            # Detect format if not specified
            if not format:
                ext = Path(file_path).suffix.lower()
                format_map = {
                    '.csv': DataFormat.CSV,
                    '.parquet': DataFormat.PARQUET,
                    '.json': DataFormat.JSON,
                    '.pkl': DataFormat.PICKLE,
                    '.pickle': DataFormat.PICKLE
                }
                format = format_map.get(ext)
                
                if not format:
                    logger.error(f"Cannot detect format for {file_path}")
                    return False
            
            # Read file
            if format == DataFormat.CSV:
                df = pd.read_csv(file_path)
            elif format == DataFormat.PARQUET:
                df = pd.read_parquet(file_path)
            elif format == DataFormat.JSON:
                df = pd.read_json(file_path)
            elif format == DataFormat.PICKLE:
                df = pd.read_pickle(file_path)
            else:
                logger.error(f"Unsupported import format: {format}")
                return False
            
            # Convert timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Store based on data type
            if data_type == DataType.OHLCV:
                if not timeframe:
                    logger.error("Timeframe required for OHLCV data")
                    return False
                await self.store_ohlcv(symbol, timeframe, df)
            else:
                logger.warning(f"Import not implemented for {data_type}")
                return False
            
            logger.info(f"Imported {len(df)} records from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {str(e)}")
            return False
    
    async def cleanup_old_data(
        self,
        data_type: Optional[DataType] = None,
        older_than_days: Optional[int] = None
    ) -> int:
        """
        Clean up old historical data
        
        Args:
            data_type: Type of data to clean (all if None)
            older_than_days: Override retention period
            
        Returns:
            Number of records deleted
        """
        try:
            total_deleted = 0
            
            # Determine data types to clean
            if data_type:
                data_types = [data_type]
            else:
                data_types = list(DataType)
            
            for dt in data_types:
                # Get retention period
                retention_days = older_than_days or self.config.retention_days.get(dt, 365)
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                # Clean database
                if self.config.use_database:
                    deleted = await self._cleanup_database(dt, cutoff_date)
                    total_deleted += deleted
                
                # Clean files
                if self.config.use_files:
                    deleted = await self._cleanup_files(dt, cutoff_date)
                    total_deleted += deleted
            
            logger.info(f"Cleaned up {total_deleted} old records")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Storage statistics
        """
        try:
            stats = {
                "database": {},
                "files": {},
                "cache": {}
            }
            
            # Database stats
            if self.config.use_database:
                async with get_db() as session:
                    # Count records
                    for table, name in [
                        (MarketData, "market_data"),
                        (SignalHistory, "signals"),
                        (AgentPerformance, "metrics")
                    ]:
                        result = await session.execute(
                            select(func.count(table.id))
                        )
                        count = result.scalar()
                        stats["database"][name] = count
            
            # File stats
            if self.config.use_files:
                total_size = 0
                file_count = 0
                
                for path in self.base_path.rglob("*"):
                    if path.is_file():
                        file_count += 1
                        total_size += path.stat().st_size
                
                stats["files"] = {
                    "count": file_count,
                    "total_size_mb": total_size / (1024 * 1024),
                    "base_path": str(self.base_path)
                }
            
            # Cache stats
            stats["cache"] = {
                "entries": len(self.cache),
                "size_mb": self.cache_size / (1024 * 1024),
                "max_size_mb": self.config.cache_size_mb
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {}
    
    async def _store_to_database(
        self,
        df: pd.DataFrame,
        data_type: DataType
    ) -> None:
        """
        Store data to database
        """
        async with get_db() as session:
            if data_type == DataType.OHLCV:
                # Prepare records
                records = df.to_dict('records')
                
                # Batch insert
                for i in range(0, len(records), self.config.database_batch_size):
                    batch = records[i:i + self.config.database_batch_size]
                    
                    for record in batch:
                        market_data = MarketData(
                            symbol=record['symbol'],
                            timeframe=record.get('timeframe', '1d'),
                            timestamp=record['timestamp'],
                            open=record['open'],
                            high=record['high'],
                            low=record['low'],
                            close=record['close'],
                            volume=record['volume'],
                            vwap=record.get('vwap'),
                            trade_count=record.get('trade_count')
                        )
                        session.add(market_data)
                    
                    await session.commit()
    
    async def _store_to_file(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: DataType,
        timeframe: Optional[TimeFrame] = None
    ) -> None:
        """
        Store data to file
        """
        # Create directory structure
        if self.config.partition_by_symbol:
            dir_path = self.base_path / symbol
        else:
            dir_path = self.base_path
        
        if self.config.partition_by_date and 'timestamp' in df.columns:
            date_str = df['timestamp'].iloc[0].strftime('%Y%m%d')
            dir_path = dir_path / date_str
        
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{data_type.value}"
        if timeframe:
            filename += f"_{timeframe.value}"
        
        # Store based on format
        if self.config.file_format == DataFormat.PARQUET:
            file_path = dir_path / f"{filename}.parquet"
            
            # Append or overwrite
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp'])
            
            df.to_parquet(file_path, compression='snappy')
            
        elif self.config.file_format == DataFormat.CSV:
            file_path = dir_path / f"{filename}.csv"
            
            # Append or create
            mode = 'a' if file_path.exists() else 'w'
            header = not file_path.exists()
            df.to_csv(file_path, mode=mode, header=header, index=False)
    
    async def _query_ohlcv(self, query: DataQuery) -> pd.DataFrame:
        """
        Query OHLCV data
        """
        # Try files first (usually faster)
        if self.config.use_files:
            dfs = []
            
            for symbol in query.symbols:
                # Find relevant files
                if self.config.partition_by_symbol:
                    search_path = self.base_path / symbol
                else:
                    search_path = self.base_path
                
                if search_path.exists():
                    for file_path in search_path.rglob(f"*{DataType.OHLCV.value}*.parquet"):
                        try:
                            df = pd.read_parquet(file_path)
                            
                            # Filter by date range
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                df = df[
                                    (df['timestamp'] >= query.start_date) &
                                    (df['timestamp'] <= query.end_date)
                                ]
                            
                            if not df.empty:
                                dfs.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to read {file_path}: {str(e)}")
            
            if dfs:
                return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['symbol', 'timestamp'])
        
        # Fall back to database
        if self.config.use_database:
            async with get_db() as session:
                stmt = select(MarketData).where(
                    and_(
                        MarketData.symbol.in_(query.symbols),
                        MarketData.timestamp >= query.start_date,
                        MarketData.timestamp <= query.end_date
                    )
                )
                
                if query.timeframe:
                    stmt = stmt.where(MarketData.timeframe == query.timeframe.value)
                
                result = await session.execute(stmt)
                records = result.scalars().all()
                
                if records:
                    data = [{
                        'symbol': r.symbol,
                        'timestamp': r.timestamp,
                        'open': r.open,
                        'high': r.high,
                        'low': r.low,
                        'close': r.close,
                        'volume': r.volume,
                        'vwap': r.vwap
                    } for r in records]
                    
                    return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    async def _query_signals(self, query: DataQuery) -> pd.DataFrame:
        """
        Query signal data
        """
        async with get_db() as session:
            stmt = select(SignalHistory).where(
                and_(
                    SignalHistory.symbol.in_(query.symbols),
                    SignalHistory.timestamp >= query.start_date,
                    SignalHistory.timestamp <= query.end_date
                )
            )
            
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            if records:
                data = [{
                    'symbol': r.symbol,
                    'timestamp': r.timestamp,
                    'agent_id': r.agent_id,
                    'signal_type': r.signal_type,
                    'confidence': r.confidence,
                    'expected_return': r.expected_return
                } for r in records]
                
                return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    async def _query_metrics(self, query: DataQuery) -> pd.DataFrame:
        """
        Query metrics data
        """
        async with get_db() as session:
            stmt = select(AgentPerformance).where(
                and_(
                    AgentPerformance.timestamp >= query.start_date,
                    AgentPerformance.timestamp <= query.end_date
                )
            )
            
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            if records:
                data = [{
                    'timestamp': r.timestamp,
                    'agent_id': r.agent_id,
                    'accuracy': r.accuracy,
                    'win_rate': r.win_rate,
                    'sharpe_ratio': r.sharpe_ratio,
                    'total_pnl': r.total_pnl
                } for r in records]
                
                return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Group by symbol
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df[mask]
            
            # SMA
            df.loc[mask, 'sma_20'] = symbol_df['close'].rolling(20).mean()
            df.loc[mask, 'sma_50'] = symbol_df['close'].rolling(50).mean()
            
            # EMA
            df.loc[mask, 'ema_12'] = symbol_df['close'].ewm(span=12).mean()
            df.loc[mask, 'ema_26'] = symbol_df['close'].ewm(span=26).mean()
            
            # RSI
            delta = symbol_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df.loc[mask, 'rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma = symbol_df['close'].rolling(20).mean()
            std = symbol_df['close'].rolling(20).std()
            df.loc[mask, 'bb_upper'] = sma + (std * 2)
            df.loc[mask, 'bb_lower'] = sma - (std * 2)
            
            # MACD
            ema_12 = symbol_df['close'].ewm(span=12).mean()
            ema_26 = symbol_df['close'].ewm(span=26).mean()
            df.loc[mask, 'macd'] = ema_12 - ema_26
            df.loc[mask, 'macd_signal'] = df.loc[mask, 'macd'].ewm(span=9).mean()
        
        return df
    
    def _merge_signals(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge signals with OHLCV data
        """
        if df.empty or signals_df.empty:
            return df
        
        # Merge on symbol and nearest timestamp
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            signals_df.sort_values('timestamp'),
            on='timestamp',
            by='symbol',
            direction='backward',
            tolerance=pd.Timedelta('1h')
        )
        
        return df
    
    def _resample_data(self, df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
        """
        Resample time series data
        """
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        df = df.set_index('timestamp')
        
        # Group by symbol and resample
        resampled_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            
            # Resample OHLCV
            resampled = symbol_df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            resampled['symbol'] = symbol
            resampled_dfs.append(resampled)
        
        if resampled_dfs:
            return pd.concat(resampled_dfs).reset_index()
        
        return pd.DataFrame()
    
    def _get_cache_key(self, query: DataQuery) -> str:
        """
        Generate cache key for query
        """
        key_parts = [
            "_".join(query.symbols),
            query.start_date.strftime('%Y%m%d'),
            query.end_date.strftime('%Y%m%d'),
            query.data_type.value
        ]
        
        if query.timeframe:
            key_parts.append(query.timeframe.value)
        
        return "_".join(key_parts)
    
    def _update_cache(self, key: str, df: pd.DataFrame) -> None:
        """
        Update cache with new data
        """
        # Estimate size
        size_bytes = df.memory_usage(deep=True).sum()
        
        # Check cache size limit
        max_size = self.config.cache_size_mb * 1024 * 1024
        
        # Evict old entries if needed
        while self.cache_size + size_bytes > max_size and self.cache:
            oldest_key = next(iter(self.cache))
            old_df, _ = self.cache.pop(oldest_key)
            self.cache_size -= old_df.memory_usage(deep=True).sum()
        
        # Add to cache
        self.cache[key] = (df.copy(), datetime.now())
        self.cache_size += size_bytes
    
    async def _append_to_signal_file(self, symbol: str, signal_data: Dict) -> None:
        """
        Append signal to file
        """
        dir_path = self.base_path / symbol / "signals"
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / f"{datetime.now().strftime('%Y%m')}.jsonl"
        
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(json.dumps(signal_data, default=str) + "\n")
    
    async def _append_to_metrics_file(self, metric_type: str, metrics: Dict) -> None:
        """
        Append metrics to file
        """
        dir_path = self.base_path / "metrics" / metric_type
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(json.dumps(metrics, default=str) + "\n")
    
    async def _cleanup_database(self, data_type: DataType, cutoff_date: datetime) -> int:
        """
        Clean up old database records
        """
        deleted = 0
        
        async with get_db() as session:
            if data_type == DataType.OHLCV:
                result = await session.execute(
                    select(MarketData).where(MarketData.timestamp < cutoff_date)
                )
                records = result.scalars().all()
                
                for record in records:
                    await session.delete(record)
                
                deleted = len(records)
                await session.commit()
            
            elif data_type == DataType.SIGNALS:
                result = await session.execute(
                    select(SignalHistory).where(SignalHistory.timestamp < cutoff_date)
                )
                records = result.scalars().all()
                
                for record in records:
                    await session.delete(record)
                
                deleted = len(records)
                await session.commit()
        
        return deleted
    
    async def _cleanup_files(self, data_type: DataType, cutoff_date: datetime) -> int:
        """
        Clean up old files
        """
        deleted = 0
        
        for file_path in self.base_path.rglob(f"*{data_type.value}*"):
            if file_path.is_file():
                # Check file modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_date:
                    file_path.unlink()
                    deleted += 1
        
        return deleted
    
    async def _flush_loop(self) -> None:
        """
        Periodically flush write buffers
        """
        while self._active:
            try:
                await asyncio.sleep(60)  # Flush every minute
                await self.flush_all()
            except Exception as e:
                logger.error(f"Flush loop error: {str(e)}")
    
    async def _cleanup_loop(self) -> None:
        """
        Periodically clean up old data
        """
        while self._active:
            try:
                await asyncio.sleep(86400)  # Clean up daily
                await self.cleanup_old_data()
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
    
    async def flush_all(self) -> None:
        """
        Flush all write buffers
        """
        async with self.buffer_lock:
            for key, buffer in self.write_buffers.items():
                if buffer:
                    # Process buffered writes
                    logger.debug(f"Flushing {len(buffer)} items for {key}")
            
            self.write_buffers.clear()


# Global historical data storage instance
historical_storage = HistoricalDataStorage()