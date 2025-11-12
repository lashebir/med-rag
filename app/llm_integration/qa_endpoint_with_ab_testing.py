# app/llm_integration/qa_endpoint_with_ab_testing.py
"""
QA Endpoint with A/B Testing Integration

Example implementation showing how to add A/B testing to your QA endpoint.
"""

import os
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import A/B testing
from app.llm_integration.ab_testing import create_ab_test_manager

load_dotenv()

# Configuration
TOP_K = int(os.getenv("TOP_K", "5"))

# Create A/B test manager (singleton)
ab_manager = create_ab_test_manager()

# ============================================================================
# Request/Response Models
# ============================================================================

class MetadataFilters(BaseModel):
    """Optional metadata filters."""
    authors: Optional[List[str]] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    institutions: Optional[List[str]] = None
    sources: Optional[List[str]] = None


class QARequestWithAB(BaseModel):
    """QA request with optional A/B testing parameters."""
    question: str = Field(..., description="User question")
    top_k: Optional[int] = Field(default=TOP_K, ge=1, le=15)
    filters: Optional[MetadataFilters] = None

    # A/B testing parameters (optional)
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for consistent strategy assignment"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking"
    )
    ab_method: Optional[str] = Field(
        default="random",
        description="Assignment method: 'random', 'consistent', or 'weighted'"
    )
    force_strategy: Optional[str] = Field(
        default=None,
        description="Force specific strategy (bypasses A/B testing): 'ivfflat_only', 'ivfflat_ner_boost', 'ivfflat_tsvector', 'ivfflat_ner_tsvector'"
    )


class UserFeedback(BaseModel):
    """User feedback on search results."""
    experiment_id: str
    clicked_chunk_ids: Optional[List[int]] = Field(
        default=None,
        description="IDs of chunks user clicked on"
    )
    user_rating: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="1-5 star rating"
    )
    user_feedback: Optional[str] = Field(
        default=None,
        description="Optional text feedback"
    )


class Citation(BaseModel):
    pmcid: str
    chunk_index: int
    source_uri: Optional[str] = None
    title: Optional[str] = None


class QAResponseWithAB(BaseModel):
    """QA response with A/B testing metadata."""
    answer: str
    citations: List[Citation]
    used_contexts: List[Dict[str, Any]]

    # A/B testing metadata
    experiment_id: str = Field(
        description="Unique experiment ID for tracking feedback"
    )
    strategy_used: str = Field(
        description="Strategy that was used for retrieval"
    )
    retrieval_latency_ms: float = Field(
        description="Time taken for retrieval in milliseconds"
    )


# ============================================================================
# Router
# ============================================================================

router = APIRouter()


@router.post("/answer", response_model=QAResponseWithAB)
def answer_with_ab_testing(req: QARequestWithAB):
    """
    Answer a question using RAG with A/B testing.

    A/B Testing Flow:
    1. User sends query
    2. System randomly assigns a search strategy
    3. Results are returned with experiment_id
    4. User interacts with results
    5. Frontend sends feedback via /feedback endpoint

    Example with A/B testing:
    ```json
    {
        "question": "What are the treatments for diabetes?",
        "top_k": 5,
        "user_id": "user_12345",
        "session_id": "session_abc",
        "ab_method": "consistent"
    }
    ```

    Example forcing specific strategy (no A/B):
    ```json
    {
        "question": "What are the treatments for diabetes?",
        "top_k": 5,
        "force_strategy": "ivfflat_ner_tsvector"
    }
    ```
    """
    start_time = time.time()

    # Step 1: Assign strategy (or use forced strategy)
    if req.force_strategy:
        # Bypass A/B testing, use specified strategy
        if req.force_strategy not in ab_manager.strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy: {req.force_strategy}. Must be one of: {list(ab_manager.strategies.keys())}"
            )
        strategy_name = req.force_strategy
        strategy_func = ab_manager.strategies[strategy_name]
        experiment_id = "forced_" + req.force_strategy
    else:
        # A/B testing: randomly assign strategy
        strategy_name, strategy_func, experiment_id = ab_manager.assign_strategy(
            user_id=req.user_id,
            session_id=req.session_id,
            method=req.ab_method
        )

    # Step 2: Execute retrieval with assigned strategy
    try:
        contexts = strategy_func(
            question=req.question,
            k=req.top_k or TOP_K
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )

    retrieval_latency_ms = (time.time() - start_time) * 1000

    if not contexts:
        raise HTTPException(
            status_code=404,
            detail="No relevant contexts found"
        )

    # Step 3: Generate answer (placeholder - integrate with your LLM)
    # For this example, we'll just return the contexts
    answer = f"Based on {len(contexts)} relevant documents about '{req.question}'..."

    # Step 4: Parse citations (simplified)
    citations = []
    for ctx in contexts[:3]:  # Top 3 as citations
        citations.append(Citation(
            pmcid=ctx["pmcid"],
            chunk_index=ctx["chunk_index"],
            source_uri=ctx.get("source_uri"),
            title=ctx.get("title"),
        ))

    # Step 5: Log interaction (without user feedback yet)
    ab_manager.log_interaction(
        experiment_id=experiment_id,
        user_id=req.user_id,
        query=req.question,
        strategy=strategy_name,
        results_count=len(contexts),
        latency_ms=retrieval_latency_ms,
        clicked_chunk_ids=[],  # Will be updated via feedback endpoint
        metadata={
            "top_k": req.top_k,
            "has_filters": req.filters is not None,
        }
    )

    return QAResponseWithAB(
        answer=answer,
        citations=citations,
        used_contexts=contexts,
        experiment_id=experiment_id,
        strategy_used=strategy_name,
        retrieval_latency_ms=retrieval_latency_ms,
    )


@router.post("/feedback")
def submit_feedback(feedback: UserFeedback):
    """
    Submit user feedback for A/B testing analysis.

    Call this endpoint when:
    - User clicks on a result
    - User provides a rating
    - User submits text feedback

    Example:
    ```json
    {
        "experiment_id": "abc-123-def",
        "clicked_chunk_ids": [12345, 67890],
        "user_rating": 4,
        "user_feedback": "Results were helpful"
    }
    ```
    """
    # Note: This is a simplified version
    # In production, you'd retrieve the original query details
    # and update the log entry

    print(f"Received feedback for experiment {feedback.experiment_id}")
    print(f"  Clicked chunks: {feedback.clicked_chunk_ids}")
    print(f"  Rating: {feedback.user_rating}")

    return {
        "status": "success",
        "message": "Feedback recorded",
        "experiment_id": feedback.experiment_id
    }


@router.get("/ab_stats")
def get_ab_stats():
    """
    Get current A/B testing statistics.

    Returns performance metrics for each strategy:
    - Total queries
    - Average latency
    - Click-through rate
    - Average user rating

    Example response:
    ```json
    {
        "ivfflat_only": {
            "total_queries": 1234,
            "avg_latency_ms": 23.4,
            "click_through_rate": 0.42,
            "avg_user_rating": 4.1
        },
        "ivfflat_ner_tsvector": {
            "total_queries": 1198,
            "avg_latency_ms": 56.7,
            "click_through_rate": 0.56,
            "avg_user_rating": 4.5
        }
    }
    ```
    """
    stats = ab_manager.get_experiment_stats()

    if not stats:
        return {
            "message": "No A/B testing data yet",
            "stats": {}
        }

    return {
        "experiment_name": ab_manager.experiment_name,
        "stats": stats
    }


@router.post("/ab_export")
def export_ab_results(output_file: str = "ab_test_results.json"):
    """
    Export A/B testing results to JSON file.

    Args:
        output_file: Name of output file

    Returns:
        Path to exported file
    """
    ab_manager.export_results(output_file)

    return {
        "status": "success",
        "file": output_file
    }


# ============================================================================
# Example: Integration with existing QA endpoint
# ============================================================================

"""
To integrate into your existing qa_endpoint.py:

1. Import A/B testing:
    from app.llm_integration.ab_testing import create_ab_test_manager
    ab_manager = create_ab_test_manager()

2. Modify your answer endpoint:
    @router.post("/answer")
    def answer(req: QARequest):
        # Assign strategy
        strategy_name, strategy_func, experiment_id = ab_manager.assign_strategy(
            user_id=req.user_id,
            method="random"
        )

        # Use assigned strategy
        contexts = strategy_func(req.question, k=req.top_k)

        # ... rest of your logic ...

        # Log interaction
        ab_manager.log_interaction(
            experiment_id=experiment_id,
            user_id=req.user_id,
            query=req.question,
            strategy=strategy_name,
            results_count=len(contexts),
            latency_ms=retrieval_latency
        )

        # Return experiment_id in response for feedback tracking
        return QAResponse(..., experiment_id=experiment_id)

3. Add feedback endpoint for user clicks/ratings
"""
