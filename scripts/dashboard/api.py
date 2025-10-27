#!/usr/bin/env python3
"""
FastAPI Server for Productivity Dashboard
==========================================
Serves aggregated metrics to the dashboard frontend via a single endpoint.

Endpoint: GET /api/data/{slice}?lookback_days=<int>&project_id=<string>
Where {slice} ∈ { "15min", "1H", "D", "W", "M" }

Returns JSON matching the dashboard template contract exactly.
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.dashboard.analytics import DashboardAnalytics

app = FastAPI(title="Productivity Dashboard API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analytics engine
analytics = DashboardAnalytics(data_dir=project_root)


@app.get("/api/data/{time_slice}")
async def get_dashboard_data(
    time_slice: str,
    lookback_days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    project_id: Optional[str] = Query("", description="Filter by project ID (empty = all projects)")
):
    """
    Get dashboard data for a specific time slice.
    
    Args:
        time_slice: One of "15min", "1H", "D", "W", "M"
        lookback_days: Number of days to include in the analysis
        project_id: Optional project ID filter
        
    Returns:
        JSON response with all dashboard data matching the template contract
    """
    # Validate time slice
    valid_slices = ["15min", "1H", "D", "W", "M"]
    if time_slice not in valid_slices:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid time_slice. Must be one of: {', '.join(valid_slices)}"
        )
    
    try:
        # Generate dashboard data using analytics engine
        data = analytics.generate_dashboard_response(
            time_slice=time_slice,
            lookback_days=lookback_days,
            project_id=project_id or None
        )
        return JSONResponse(content=data)
    
    except Exception as e:
        print(f"Error generating dashboard data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate dashboard data: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "productivity-dashboard-api"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Productivity Dashboard API...")
    print(f"📂 Data directory: {project_root}")
    print("🌐 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


