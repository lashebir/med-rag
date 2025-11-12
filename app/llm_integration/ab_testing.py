# app/llm_integration/ab_testing.py
"""
A/B Testing Module for Search Strategies

Implements random strategy assignment and result tracking for production A/B tests.
Allows comparing different retrieval strategies based on real user behavior.

Usage:
    from app.llm_integration.ab_testing import ABTestManager

    ab_manager = ABTestManager()
    strategy, experiment_id = ab_manager.assign_strategy(user_id="user123")

    # Execute search with assigned strategy
    results = strategy(question, k=10)

    # Track user interaction
    ab_manager.log_interaction(
        experiment_id=experiment_id,
        user_id="user123",
        query=question,
        clicked_chunk_ids=[results[0]['chunk_id']],
        user_rating=4
    )
"""

import os
import json
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dotenv import load_dotenv
from psycopg import connect

load_dotenv()

PG_KWARGS = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "medrag"),
    user=os.getenv("PGUSER", os.getenv("USER")),
    password=os.getenv("PGPASSWORD") or None,
)


class ABTestManager:
    """
    Manages A/B testing experiments for search strategies.

    Handles:
    - Random strategy assignment
    - Result logging
    - Statistical analysis
    """

    def __init__(
        self,
        experiment_name: str = "search_strategy_comparison",
        log_dir: str = "ab_test_logs"
    ):
        """
        Initialize A/B test manager.

        Args:
            experiment_name: Name of the current experiment
            log_dir: Directory to store experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Strategy registry
        self.strategies: Dict[str, Callable] = {}

        # Strategy weights (for weighted random assignment)
        self.strategy_weights: Dict[str, float] = {}

        # Load existing experiment state
        self.experiment_file = self.log_dir / f"{experiment_name}.jsonl"

    def register_strategy(
        self,
        name: str,
        strategy_func: Callable,
        weight: float = 1.0
    ):
        """
        Register a search strategy for A/B testing.

        Args:
            name: Strategy identifier (e.g., "ivfflat_only")
            strategy_func: Function that executes the strategy
            weight: Probability weight (higher = more likely to be selected)
        """
        self.strategies[name] = strategy_func
        self.strategy_weights[name] = weight

    def assign_strategy(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        method: str = "random"
    ) -> Tuple[str, Callable, str]:
        """
        Assign a search strategy to a user/session.

        Args:
            user_id: User identifier (for consistent assignment)
            session_id: Session identifier
            method: Assignment method:
                    - "random": Random assignment each time
                    - "consistent": Same user gets same strategy
                    - "weighted": Weighted random based on strategy_weights

        Returns:
            Tuple of (strategy_name, strategy_function, experiment_id)
        """
        if not self.strategies:
            raise ValueError("No strategies registered. Use register_strategy() first.")

        # Generate experiment ID
        experiment_id = self._generate_experiment_id(user_id, session_id)

        # Assign strategy based on method
        if method == "consistent" and user_id:
            # Consistent hashing: same user always gets same strategy
            strategy_name = self._hash_assign(user_id)
        elif method == "weighted":
            # Weighted random assignment
            strategy_name = self._weighted_random()
        else:
            # Pure random assignment
            strategy_name = random.choice(list(self.strategies.keys()))

        strategy_func = self.strategies[strategy_name]

        # Log assignment
        self._log_event({
            "event": "strategy_assigned",
            "experiment_id": experiment_id,
            "user_id": user_id,
            "session_id": session_id,
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
        })

        return strategy_name, strategy_func, experiment_id

    def log_interaction(
        self,
        experiment_id: str,
        user_id: Optional[str],
        query: str,
        strategy: str,
        results_count: int,
        latency_ms: float,
        clicked_chunk_ids: Optional[List[int]] = None,
        user_rating: Optional[int] = None,
        user_feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a user interaction with search results.

        Args:
            experiment_id: Unique experiment identifier
            user_id: User identifier
            query: User's search query
            strategy: Strategy that was used
            results_count: Number of results returned
            latency_ms: Query latency in milliseconds
            clicked_chunk_ids: IDs of chunks user clicked on
            user_rating: Optional 1-5 star rating
            user_feedback: Optional text feedback
            metadata: Additional metadata
        """
        event = {
            "event": "user_interaction",
            "experiment_id": experiment_id,
            "user_id": user_id,
            "query": query,
            "strategy": strategy,
            "results_count": results_count,
            "latency_ms": latency_ms,
            "clicked_chunk_ids": clicked_chunk_ids or [],
            "user_rating": user_rating,
            "user_feedback": user_feedback,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self._log_event(event)

    def get_experiment_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current experiment.

        Returns:
            Dictionary with stats per strategy:
            - total_queries
            - avg_latency_ms
            - click_through_rate
            - avg_user_rating
        """
        if not self.experiment_file.exists():
            return {}

        # Read all events
        events = []
        with open(self.experiment_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))

        # Group by strategy
        stats_by_strategy: Dict[str, Dict] = {}

        for event in events:
            if event["event"] != "user_interaction":
                continue

            strategy = event["strategy"]

            if strategy not in stats_by_strategy:
                stats_by_strategy[strategy] = {
                    "total_queries": 0,
                    "total_latency_ms": 0,
                    "total_clicks": 0,
                    "total_ratings": 0,
                    "sum_ratings": 0,
                }

            stats = stats_by_strategy[strategy]
            stats["total_queries"] += 1
            stats["total_latency_ms"] += event["latency_ms"]
            stats["total_clicks"] += len(event["clicked_chunk_ids"])

            if event.get("user_rating"):
                stats["total_ratings"] += 1
                stats["sum_ratings"] += event["user_rating"]

        # Calculate averages
        results = {}
        for strategy, stats in stats_by_strategy.items():
            results[strategy] = {
                "total_queries": stats["total_queries"],
                "avg_latency_ms": stats["total_latency_ms"] / stats["total_queries"],
                "click_through_rate": stats["total_clicks"] / stats["total_queries"],
                "avg_user_rating": (
                    stats["sum_ratings"] / stats["total_ratings"]
                    if stats["total_ratings"] > 0 else None
                ),
            }

        return results

    def export_results(self, output_file: str):
        """
        Export experiment results to JSON for analysis.

        Args:
            output_file: Path to output JSON file
        """
        stats = self.get_experiment_stats()

        with open(output_file, 'w') as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
            }, f, indent=2)

        print(f"Experiment results exported to: {output_file}")

    def _generate_experiment_id(
        self,
        user_id: Optional[str],
        session_id: Optional[str]
    ) -> str:
        """Generate unique experiment ID."""
        import uuid
        return str(uuid.uuid4())

    def _hash_assign(self, user_id: str) -> str:
        """Consistent hash-based strategy assignment."""
        # Hash user_id to consistently assign same strategy
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        strategy_list = list(self.strategies.keys())
        index = hash_val % len(strategy_list)
        return strategy_list[index]

    def _weighted_random(self) -> str:
        """Weighted random strategy selection."""
        strategies = list(self.strategy_weights.keys())
        weights = [self.strategy_weights[s] for s in strategies]
        return random.choices(strategies, weights=weights, k=1)[0]

    def _log_event(self, event: Dict[str, Any]):
        """Log event to JSONL file."""
        with open(self.experiment_file, 'a') as f:
            f.write(json.dumps(event) + '\n')


# ============================================================================
# Helper function to create configured AB test manager
# ============================================================================

def create_ab_test_manager() -> ABTestManager:
    """
    Create and configure AB test manager with all strategies.

    Returns:
        Configured ABTestManager instance
    """
    from app.search_types.search_strategies import (
        retrieve_ivfflat_only,
        retrieve_ivfflat_ner_boost,
        retrieve_ivfflat_tsvector,
        retrieve_ivfflat_ner_tsvector,
    )

    manager = ABTestManager(experiment_name="search_strategy_comparison_v1")

    # Register all strategies with equal weight
    manager.register_strategy("ivfflat_only", retrieve_ivfflat_only, weight=1.0)
    manager.register_strategy("ivfflat_ner_boost", retrieve_ivfflat_ner_boost, weight=1.0)
    manager.register_strategy("ivfflat_tsvector", retrieve_ivfflat_tsvector, weight=1.0)
    manager.register_strategy("ivfflat_ner_tsvector", retrieve_ivfflat_ner_tsvector, weight=1.0)

    return manager
