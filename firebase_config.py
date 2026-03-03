"""
Firebase configuration and state management for the trading system.
Handles persistence of optimization results, strategy states, and market data.
Architectural Choice: Using Firestore for its real-time capabilities and 
scalability, avoiding SQL databases that require schema migrations.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

logger = logging.getLogger(__name__)

class TradingState(Enum):
    """Trading strategy operational states"""
    INITIALIZING = "initializing"
    OPTIMIZING = "optimizing"
    TRADING = "trading"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class OptimizationResult:
    """Structured storage for optimization results"""
    timestamp: datetime
    strategy_name: str
    optimized_params: Dict[str, float]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    backtest_period_days: int
    sample_size: int
    
@dataclass
class TradeRecord:
    """Structured trade record for audit trail"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    strategy_params: Dict[str, float]
    pnl: Optional[float] = None
    status: str = "executed"

class FirebaseManager:
    """Manages all Firestore operations with error handling and retry logic"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Firebase connection with graceful fallback
        
        Args:
            config_path: Path to service account key JSON file
        """
        self.db = None
        self._initialized = False
        
        try:
            # Try environment variable first
            cred_path = config_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            if not cred_path:
                logger.warning("Firebase credentials not found. Running in local mode.")
                return
            
            if not os.path.exists(cred_path):
                logger.error(f"Credential file not found at {cred_path}")
                return
            
            # Initialize Firebase app (handles multiple initializations)
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            self._initialized = True
            logger.info("Firebase Firestore initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            self._initialized = False
    
    def is_available(self) -> bool:
        """Check if Firestore is available"""
        return self._initialized and self.db is not None
    
    def save_optimization_result(self, result: OptimizationResult) -> bool:
        """
        Save optimization result to Firestore with transaction safety
        
        Args:
            result: OptimizationResult dataclass
            
        Returns:
            bool: Success status
        """
        if not self.is_available():
            logger.warning("Firestore not available, using local storage")
            return self._save_local(result)
        
        try:
            doc_ref = self.db.collection('optimization_results').document()
            doc_ref.set({
                **asdict(result),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'firestore_id': doc_ref.id
            })
            logger.info(f"Saved optimization result: {result.strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save optimization result: {str(e)}")
            return False
    
    def get_latest_optimization(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve latest optimization result for a strategy
        
        Args:
            strategy_name: Name of trading strategy
            
        Returns:
            Optional[Dict]: Latest optimization parameters or None
        """
        if not self.is_available():
            return self._load_local_latest(strategy_name)
        
        try:
            query = (self.db.collection('optimization_results')
                    .where(filter=FieldFilter('strategy_name', '==', strategy_name))
                    .order_by('timestamp', direction=firestore.Query.DESCENDING)
                    .limit(1))
            
            docs = query.get()
            
            if docs:
                return docs[0].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch optimization result: {str(e)}")
            return None
    
    def save_trade_record(self, trade: TradeRecord) -> bool:
        """
        Save trade record with atomic transaction
        
        Args:
            trade: TradeRecord dataclass
            
        Returns:
            bool: Success status
        """
        if not self.is_available():
            return self._save_local_trade(trade)
        
        try:
            @firestore.transactional
            def transactional_save(transaction, trade_ref, trade_data):
                transaction.set(trade_ref, trade_data)
            
            transaction = self.db.transaction()
            trade_ref = self.db.collection('trades').document(trade.trade_id)
            transactional_save(transaction, trade_ref, asdict(trade))
            
            logger.info(f"Saved trade record: {trade.trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save trade record: {str(e)}")
            return False
    
    def update_strategy_state(self, strategy_id: