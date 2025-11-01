#!/usr/bin/env python3
"""
Query Snapshots with DuckDB
============================
Provides a SQL interface to query snapshot data using DuckDB.

DuckDB can query JSONL files directly without importing to a database.

Usage:
    # Interactive mode
    python scripts/query_snapshots_duckdb.py

    # Execute query
    python scripts/query_snapshots_duckdb.py --query "SELECT script_id, COUNT(*) FROM events GROUP BY script_id"

    # Export to CSV
    python scripts/query_snapshots_duckdb.py --query "..." --output results.csv

Requirements:
    pip install duckdb

Note: This script requires duckdb to be installed. Install with:
    pip install duckdb
"""

import argparse
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("❌ DuckDB not installed. Install with: pip install duckdb")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "snapshot"


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create views for all snapshot datasets."""
    # Operation events view
    events_pattern = str(
        SNAPSHOT_DIR / "operation_events_v1" / "day=*" / "events.jsonl"
    )
    con.execute(f"""
        CREATE OR REPLACE VIEW events AS
        SELECT * FROM read_json_auto('{events_pattern}', union_by_name=true)
    """)
    print("  ✓ Created view: events")

    # Derived sessions view
    derived_sessions_pattern = str(
        SNAPSHOT_DIR / "derived_sessions_v1" / "day=*" / "sessions.jsonl"
    )
    con.execute(f"""
        CREATE OR REPLACE VIEW derived_sessions AS
        SELECT * FROM read_json_auto('{derived_sessions_pattern}', union_by_name=true)
    """)
    print("  ✓ Created view: derived_sessions")

    # Legacy timer sessions view
    timer_sessions_pattern = str(
        SNAPSHOT_DIR / "timer_sessions_v1" / "day=*" / "sessions.jsonl"
    )
    con.execute(f"""
        CREATE OR REPLACE VIEW timer_sessions AS
        SELECT * FROM read_json_auto('{timer_sessions_pattern}', union_by_name=true)
    """)
    print("  ✓ Created view: timer_sessions")

    # Progress snapshots view
    progress_pattern = str(
        SNAPSHOT_DIR / "progress_snapshots_v1" / "day=*" / "snapshots.jsonl"
    )
    con.execute(f"""
        CREATE OR REPLACE VIEW progress_snapshots AS
        SELECT * FROM read_json_auto('{progress_pattern}', union_by_name=true)
    """)
    print("  ✓ Created view: progress_snapshots")

    # Projects view
    projects_file = str(SNAPSHOT_DIR / "projects_v1" / "projects.jsonl")
    con.execute(f"""
        CREATE OR REPLACE VIEW projects AS
        SELECT * FROM read_json_auto('{projects_file}', union_by_name=true)
    """)
    print("  ✓ Created view: projects")

    # Daily aggregates view
    aggregates_pattern = str(
        SNAPSHOT_DIR / "daily_aggregates_v1" / "day=*" / "aggregate.json"
    )
    con.execute(f"""
        CREATE OR REPLACE VIEW daily_aggregates AS
        SELECT * FROM read_json_auto('{aggregates_pattern}', union_by_name=true)
    """)
    print("  ✓ Created view: daily_aggregates")


def show_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Show available views and their schemas."""
    print("\n📊 Available Views:")
    print("=" * 70)

    views = [
        "events",
        "derived_sessions",
        "timer_sessions",
        "progress_snapshots",
        "projects",
        "daily_aggregates",
    ]

    for view in views:
        try:
            result = con.execute(f"DESCRIBE {view}").fetchall()
            print(f"\n{view}:")
            for col_name, col_type, null, key, default, extra in result:
                print(f"  - {col_name}: {col_type}")
        except Exception as e:
            print(f"\n{view}: ⚠️  Not available ({e})")


def execute_query(
    con: duckdb.DuckDBPyConnection, query: str, output_file: str = None
) -> None:
    """Execute a SQL query and display/save results."""
    try:
        result = con.execute(query)

        if output_file:
            # Export to CSV
            con.execute(f"COPY ({query}) TO '{output_file}' (HEADER, DELIMITER ',')")
            print(f"✅ Results exported to: {output_file}")
        else:
            # Display results
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]

            print("\n" + "=" * 70)
            print("Query Results")
            print("=" * 70)
            print(" | ".join(columns))
            print("-" * 70)

            for row in rows:
                print(" | ".join(str(v) for v in row))

            print(f"\n{len(rows)} rows")

    except Exception as e:
        print(f"❌ Query error: {e}")


def interactive_mode(con: duckdb.DuckDBPyConnection) -> None:
    """Run in interactive SQL mode."""
    print("\n" + "=" * 70)
    print("DuckDB Interactive Query Interface")
    print("=" * 70)
    print()
    print("Available views:")
    print("  - events (operation events)")
    print("  - derived_sessions (derived sessions)")
    print("  - timer_sessions (legacy timer sessions)")
    print("  - progress_snapshots (crop/sorter progress)")
    print("  - projects (project manifests)")
    print("  - daily_aggregates (pre-aggregated daily stats)")
    print()
    print("Commands:")
    print("  .schema [view]  - Show schema for a view")
    print("  .tables         - List all views")
    print("  .quit or exit   - Exit")
    print()
    print("Enter SQL queries and press Enter to execute.")
    print("=" * 70)
    print()

    while True:
        try:
            query = input("duckdb> ").strip()

            if not query:
                continue

            if query.lower() in [".quit", "exit", "quit"]:
                print("Goodbye!")
                break

            if query.lower() == ".tables":
                result = con.execute("SHOW TABLES").fetchall()
                print("\nViews:")
                for row in result:
                    print(f"  - {row[0]}")
                print()
                continue

            if query.lower().startswith(".schema"):
                parts = query.split()
                view_name = parts[1] if len(parts) > 1 else None

                if view_name:
                    result = con.execute(f"DESCRIBE {view_name}").fetchall()
                    print(f"\nSchema for {view_name}:")
                    for col_name, col_type, null, key, default, extra in result:
                        print(f"  {col_name}: {col_type}")
                else:
                    show_schema(con)
                print()
                continue

            # Execute query
            result = con.execute(query)
            rows = result.fetchall()
            columns = [desc[0] for desc in result.description]

            print()
            print(" | ".join(columns))
            print("-" * 70)

            for row in rows[:100]:  # Limit to first 100 rows
                print(" | ".join(str(v) for v in row))

            if len(rows) > 100:
                print(f"... ({len(rows) - 100} more rows)")

            print(f"\n{len(rows)} rows")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query snapshot data using DuckDB SQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/query_snapshots_duckdb.py
  
  # Execute query
  python scripts/query_snapshots_duckdb.py --query "SELECT script_id, COUNT(*) FROM events GROUP BY script_id ORDER BY COUNT(*) DESC"
  
  # Export to CSV
  python scripts/query_snapshots_duckdb.py --query "SELECT * FROM projects" --output projects.csv
  
  # Show schema
  python scripts/query_snapshots_duckdb.py --show-schema

Sample Queries:
  # Top scripts by event count
  SELECT script_id, COUNT(*) as count FROM events GROUP BY script_id ORDER BY count DESC LIMIT 10
  
  # Daily file processing totals
  SELECT day, SUM(files_processed) as total_files FROM derived_sessions GROUP BY day ORDER BY day
  
  # Active projects
  SELECT project_id, title, status, initial_images FROM projects WHERE status = 'active'
  
  # Operations by type
  SELECT operation, COUNT(*) as count FROM events WHERE operation IS NOT NULL GROUP BY operation ORDER BY count DESC
        """,
    )

    parser.add_argument("--query", help="SQL query to execute")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument(
        "--show-schema", action="store_true", help="Show schema for all views"
    )

    args = parser.parse_args()

    # Initialize DuckDB
    print("Initializing DuckDB...")
    con = duckdb.connect(database=":memory:")

    # Create views
    print("\nCreating views...")
    try:
        create_views(con)
    except Exception as e:
        print(f"❌ Error creating views: {e}")
        sys.exit(1)

    # Execute based on arguments
    if args.show_schema:
        show_schema(con)
    elif args.query:
        execute_query(con, args.query, args.output)
    else:
        interactive_mode(con)


if __name__ == "__main__":
    main()
