# A/B Testing Guide for Search Strategies

## Overview

This guide explains how to run **production A/B tests** to determine which search strategy works best for your users.

## Benchmarking vs A/B Testing

| | **Benchmarking** | **A/B Testing** |
|---|---|---|
| **What** | Offline evaluation | Production testing |
| **Where** | `benchmark_search_strategies.py` | QA endpoint |
| **Metrics** | Latency, recall, overlap | User clicks, ratings, satisfaction |
| **Users** | Synthetic queries | Real users |
| **Purpose** | Technical comparison | Business decision |

**Use benchmarking** to narrow down candidates → **Use A/B testing** to pick the winner!

## Architecture

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       v
┌─────────────────────────────┐
│  A/B Test Manager           │
│  Randomly assigns strategy: │
│  - ivfflat_only (25%)      │
│  - ivfflat_ner_boost (25%) │
│  - ivfflat_tsvector (25%)  │
│  - ivfflat_ner_tsvector(25%)│
└─────────────┬───────────────┘
              │
              v
┌──────────────────────────────┐
│  Execute assigned strategy   │
│  Return results + exp_id     │
└──────────────┬───────────────┘
               │
               v
┌──────────────────────────────┐
│  User interacts with results │
│  - Clicks on chunks          │
│  - Provides rating           │
└──────────────┬───────────────┘
               │
               v
┌──────────────────────────────┐
│  Log interaction data        │
│  - experiment_id             │
│  - clicked_chunk_ids         │
│  - user_rating               │
└──────────────────────────────┘
               │
               v
┌──────────────────────────────┐
│  Analyze results             │
│  Which strategy performed    │
│  best for real users?        │
└──────────────────────────────┘
```

## Implementation

### 1. Basic Setup

```python
# In your main.py or qa_endpoint.py
from app.llm_integration.ab_testing import create_ab_test_manager

# Create AB test manager (do this once at startup)
ab_manager = create_ab_test_manager()
```

### 2. Integrate into QA Endpoint

```python
from fastapi import APIRouter
from app.llm_integration.ab_testing import create_ab_test_manager
import time

router = APIRouter()
ab_manager = create_ab_test_manager()

@router.post("/answer")
def answer(request: QARequest):
    # Step 1: Assign strategy
    strategy_name, strategy_func, experiment_id = ab_manager.assign_strategy(
        user_id=request.user_id,      # Optional: for consistent assignment
        session_id=request.session_id, # Optional: for tracking
        method="random"                # or "consistent" or "weighted"
    )

    # Step 2: Execute search
    start_time = time.time()
    contexts = strategy_func(request.question, k=request.top_k)
    latency_ms = (time.time() - start_time) * 1000

    # Step 3: Generate answer (your existing logic)
    answer = generate_answer(request.question, contexts)

    # Step 4: Log interaction
    ab_manager.log_interaction(
        experiment_id=experiment_id,
        user_id=request.user_id,
        query=request.question,
        strategy=strategy_name,
        results_count=len(contexts),
        latency_ms=latency_ms
    )

    # Step 5: Return results with experiment_id
    return {
        "answer": answer,
        "contexts": contexts,
        "experiment_id": experiment_id,  # IMPORTANT: Return this!
        "strategy": strategy_name          # Optional: for debugging
    }
```

### 3. Add Feedback Endpoint

```python
@router.post("/feedback")
def submit_feedback(
    experiment_id: str,
    clicked_chunk_ids: List[int] = None,
    user_rating: int = None,
    user_feedback: str = None
):
    """
    Users submit feedback after viewing results.

    Call this from your frontend when:
    - User clicks on a result
    - User rates the results
    - User provides text feedback
    """
    # This would typically update the existing log entry
    # For now, we'll create a new log entry
    ab_manager.log_interaction(
        experiment_id=experiment_id,
        user_id=None,  # Already logged in initial interaction
        query="",       # Already logged
        strategy="",    # Already logged
        results_count=0,
        latency_ms=0,
        clicked_chunk_ids=clicked_chunk_ids,
        user_rating=user_rating,
        user_feedback=user_feedback
    )

    return {"status": "success"}
```

### 4. Frontend Integration (JavaScript)

```javascript
// When user submits a query
async function searchQuery(question) {
  const response = await fetch('/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: question,
      user_id: getCurrentUserId(),  // Get from session
      top_k: 10
    })
  });

  const data = await response.json();

  // IMPORTANT: Store experiment_id for feedback
  currentExperimentId = data.experiment_id;

  // Display results
  displayResults(data.contexts);
}

// When user clicks on a result
function onResultClick(chunkId) {
  clickedChunks.push(chunkId);

  // Send feedback immediately or on page unload
  sendFeedback();
}

// When user provides rating
function onRatingSubmit(rating) {
  sendFeedback(rating);
}

async function sendFeedback(rating = null) {
  await fetch('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      experiment_id: currentExperimentId,
      clicked_chunk_ids: clickedChunks,
      user_rating: rating
    })
  });
}
```

## Assignment Methods

### 1. Random Assignment (Default)

```python
strategy_name, strategy_func, exp_id = ab_manager.assign_strategy(
    method="random"
)
```

- **Pro**: True randomization, unbiased
- **Con**: Same user might get different strategies
- **Use when**: Quick experiments, large user base

### 2. Consistent Assignment (Sticky)

```python
strategy_name, strategy_func, exp_id = ab_manager.assign_strategy(
    user_id="user_12345",
    method="consistent"
)
```

- **Pro**: Same user always gets same strategy
- **Con**: Requires user_id
- **Use when**: Want consistent user experience

### 3. Weighted Assignment

```python
# Configure weights when creating manager
ab_manager.register_strategy("ivfflat_ner_tsvector", func, weight=2.0)  # 2x more likely
ab_manager.register_strategy("ivfflat_only", func, weight=1.0)

strategy_name, strategy_func, exp_id = ab_manager.assign_strategy(
    method="weighted"
)
```

- **Pro**: Can favor promising strategies
- **Con**: Biased distribution
- **Use when**: Multi-armed bandit approach

## Metrics to Track

### 1. Click-Through Rate (CTR)

```
CTR = (Number of clicked results) / (Total queries)
```

**High CTR = Users find results relevant**

### 2. Average User Rating

```
Avg Rating = Sum of all ratings / Number of ratings
```

**Higher = Better user satisfaction**

### 3. Query Latency

```
Avg Latency = Sum of all latencies / Number of queries
```

**Lower = Faster, better UX**

### 4. Top-1 Click Rate

```
Top-1 Rate = (Queries where user clicked first result) / (Total queries)
```

**High = First result is very relevant**

## Analyzing Results

### View Current Stats

```python
# Add endpoint
@router.get("/ab_stats")
def get_ab_stats():
    return ab_manager.get_experiment_stats()
```

**Example output:**
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

### Statistical Significance

Before declaring a winner, ensure statistical significance:

```python
from scipy import stats

def is_significant(strategy_a_ctr, strategy_b_ctr, n_a, n_b, alpha=0.05):
    """
    Test if difference in CTR is statistically significant.

    Args:
        strategy_a_ctr: CTR for strategy A
        strategy_b_ctr: CTR for strategy B
        n_a: Number of queries for A
        n_b: Number of queries for B
        alpha: Significance level (0.05 = 95% confidence)

    Returns:
        bool: True if significant
    """
    # Two-proportion z-test
    pooled_ctr = (strategy_a_ctr * n_a + strategy_b_ctr * n_b) / (n_a + n_b)
    se = (pooled_ctr * (1 - pooled_ctr) * (1/n_a + 1/n_b)) ** 0.5
    z_score = (strategy_a_ctr - strategy_b_ctr) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return p_value < alpha

# Example
ctr_full_hybrid = 0.56
ctr_baseline = 0.42
n_full = 1198
n_baseline = 1234

if is_significant(ctr_full_hybrid, ctr_baseline, n_full, n_baseline):
    print("Full hybrid strategy is significantly better!")
```

## Sample Size Requirements

**Rule of thumb**: Need at least 100 queries per strategy to detect meaningful differences.

For 4 strategies:
- **Minimum**: 400 total queries (100 each)
- **Recommended**: 2000+ total queries (500+ each)

## Decision Framework

Choose the winning strategy based on:

1. **Primary metric**: Click-through rate (CTR)
2. **Secondary metrics**:
   - User ratings
   - Top-1 click rate
3. **Constraint**: Latency must be acceptable (<100ms)

**Example decision:**

```
Strategy             | CTR  | Rating | Latency | Decision
---------------------|------|--------|---------|----------
ivfflat_only         | 0.42 | 4.1    | 23ms    | ❌ Low CTR
ivfflat_ner_boost    | 0.51 | 4.3    | 34ms    | ✅ Good
ivfflat_tsvector     | 0.49 | 4.2    | 45ms    | ✅ OK
ivfflat_ner_tsvector | 0.56 | 4.5    | 57ms    | 🏆 WINNER
```

**Winner**: `ivfflat_ner_tsvector`
- Highest CTR (0.56)
- Highest rating (4.5)
- Acceptable latency (<100ms)

## Best Practices

### 1. Run Experiment for Sufficient Time

- **Minimum**: 1 week
- **Recommended**: 2-4 weeks
- **Why**: Account for day-of-week effects, user behavior changes

### 2. Monitor Continuously

```bash
# Check stats daily
curl http://localhost:8000/ab_stats

# Export results weekly
curl -X POST http://localhost:8000/ab_export
```

### 3. Set Success Criteria Before Starting

Define in advance:
- What CTR difference is meaningful? (e.g., +10%)
- What latency is acceptable? (<100ms)
- Minimum sample size (e.g., 500 queries per strategy)

### 4. Avoid Peeking Too Early

Don't stop experiment early even if one strategy appears to be winning. Wait for:
- Minimum sample size
- Statistical significance
- Minimum time period

## Complete Workflow Example

### Phase 1: Setup (Day 1)

1. Integrate A/B testing into QA endpoint
2. Add feedback endpoint
3. Update frontend to track clicks and ratings
4. Deploy to production

### Phase 2: Data Collection (Weeks 1-2)

1. Users query naturally
2. System randomly assigns strategies
3. Logs accumulate in `ab_test_logs/`

### Phase 3: Analysis (Week 3)

```bash
# Get current stats
curl http://localhost:8000/ab_stats

# Export for detailed analysis
curl -X POST http://localhost:8000/ab_export
python analyze_ab_results.py ab_test_results.json
```

### Phase 4: Decision (Week 3)

1. Review metrics
2. Check statistical significance
3. Choose winner
4. Deploy winning strategy to 100% of users

### Phase 5: Rollout (Week 4)

```python
# Update endpoint to use winning strategy by default
@router.post("/answer")
def answer(request: QARequest):
    # No more A/B testing - use winner
    strategy_func = retrieve_ivfflat_ner_tsvector
    contexts = strategy_func(request.question, k=request.top_k)
    # ...
```

## Troubleshooting

### "Not enough data to analyze"

**Solution**: Run experiment longer, need at least 100 queries per strategy

### "Results are inconsistent day-to-day"

**Solution**: Normal variance. Need more data or longer experiment.

### "All strategies perform similarly"

**Solution**: Good problem! Any strategy will work. Choose fastest one.

### "Latency varies wildly"

**Solution**: Cold start effect. Consider adding warmup queries or monitoring separately.

## Files Created

- `app/llm_integration/ab_testing.py` - A/B testing manager
- `app/llm_integration/qa_endpoint_with_ab_testing.py` - Example integration
- `app/AB_TESTING_GUIDE.md` - This guide
- `ab_test_logs/` - Experiment logs (created automatically)

## Summary

**Benchmarking** (`benchmark_search_strategies.py`):
- ✅ Use for initial comparison
- ✅ Technical metrics (latency, recall)
- ❌ Not real user behavior

**A/B Testing** (QA endpoint + ab_testing.py):
- ✅ Real production data
- ✅ User behavior metrics (CTR, ratings)
- ✅ Statistically rigorous
- 🎯 **Use this to make final decision**

**Next Steps:**
1. Integrate A/B testing into your QA endpoint
2. Deploy to production
3. Collect 2+ weeks of data
4. Analyze results and choose winner
5. Roll out winning strategy
