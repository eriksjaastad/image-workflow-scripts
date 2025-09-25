#!/usr/bin/env python3
"""
Demo Timer Data Generator
=========================

Generates realistic sample data to demonstrate what the activity timer
analytics will look like with real usage patterns.

This shows you exactly what insights you'll get from the timer system.
"""

import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.activity_timer import TimerReporter

def generate_realistic_session(script_name, date, start_hour, duration_minutes, files_count):
    """Generate a realistic session with natural patterns"""
    
    # Base timestamp for the date
    base_date = datetime.strptime(date, '%Y%m%d')
    start_time = base_date.replace(hour=start_hour, minute=random.randint(0, 59))
    start_timestamp = start_time.timestamp()
    
    # Calculate realistic efficiency based on time of day and duration
    if 9 <= start_hour <= 11:  # Morning peak
        base_efficiency = random.uniform(0.85, 0.95)
    elif 14 <= start_hour <= 16:  # Afternoon peak  
        base_efficiency = random.uniform(0.80, 0.90)
    elif 11 <= start_hour <= 13:  # Pre-lunch
        base_efficiency = random.uniform(0.75, 0.85)
    else:  # Other times
        base_efficiency = random.uniform(0.70, 0.80)
        
    # Adjust efficiency based on session length (fatigue factor)
    if duration_minutes > 120:  # 2+ hours
        base_efficiency *= 0.9
    elif duration_minutes > 180:  # 3+ hours
        base_efficiency *= 0.8
        
    # Calculate times
    total_time = duration_minutes * 60
    active_time = total_time * base_efficiency
    idle_time = total_time - active_time
    end_time = start_timestamp + total_time
    
    # Generate operations based on script type
    operations = []
    batches = []
    
    if script_name == "01_web_image_selector":
        # Image selection operations
        selections = int(files_count * 0.6)  # 60% kept
        deletions = files_count - selections
        
        operations.extend([
            {"type": "select", "file_count": selections, "timestamp": start_time.isoformat()},
            {"type": "delete", "file_count": deletions, "timestamp": start_time.isoformat()},
            {"type": "move", "file_count": selections, "timestamp": start_time.isoformat()}
        ])
        
        # Generate batches (every ~100 files)
        batch_count = max(1, files_count // 100)
        for i in range(batch_count):
            batch_start = start_timestamp + (i * total_time / batch_count)
            batch_end = start_timestamp + ((i + 1) * total_time / batch_count)
            batches.append({
                "name": f"Image Selection Batch {i+1}",
                "start_time": batch_start,
                "end_time": batch_end,
                "duration": batch_end - batch_start
            })
            
    elif script_name == "04_batch_crop_tool":
        # Cropping operations
        operations.append({
            "type": "crop", 
            "file_count": files_count, 
            "timestamp": start_time.isoformat()
        })
        
        # Batch every ~50 crops (3-image batches)
        batch_count = max(1, files_count // 50)
        for i in range(batch_count):
            batch_start = start_timestamp + (i * total_time / batch_count)
            batch_end = start_timestamp + ((i + 1) * total_time / batch_count)
            batches.append({
                "name": f"Crop Batch {i+1}",
                "start_time": batch_start,
                "end_time": batch_end,
                "duration": batch_end - batch_start
            })
            
    elif script_name == "03_web_character_sorter":
        # Character sorting operations
        operations.append({
            "type": "sort", 
            "file_count": files_count, 
            "timestamp": start_time.isoformat()
        })
        
        # Usually one batch per session
        batches.append({
            "name": "Character Sorting Session",
            "start_time": start_timestamp,
            "end_time": end_time,
            "duration": total_time
        })
    
    # Create session data
    session_id = start_time.strftime("%Y%m%d_%H%M%S")
    
    return {
        "script_name": script_name,
        "session_id": session_id,
        "start_time": start_timestamp,
        "end_time": end_time,
        "active_time": active_time,
        "idle_time": idle_time,
        "last_activity": end_time,
        "batches": batches,
        "operations": operations
    }

def generate_demo_data(days=14):
    """Generate demo data for the last N days"""
    
    # Create timer data directory
    data_dir = Path(__file__).parent / "timer_data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"🎭 Generating {days} days of realistic timer data...")
    
    # Script usage patterns (realistic workflow)
    scripts = [
        ("01_web_image_selector", 0.4, 120, 800),    # 40% of time, 2h avg, 800 files avg
        ("04_batch_crop_tool", 0.35, 90, 200),       # 35% of time, 1.5h avg, 200 files avg  
        ("03_web_character_sorter", 0.15, 45, 150),  # 15% of time, 45m avg, 150 files avg
        ("05_web_multi_directory_viewer", 0.1, 30, 100)  # 10% of time, 30m avg, 100 files avg
    ]
    
    for day_offset in range(days):
        date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y%m%d')
        daily_sessions = []
        
        # Simulate realistic daily usage (3-6 sessions per day)
        session_count = random.randint(2, 5)
        
        # Distribute sessions throughout work day (9am-6pm)
        work_hours = list(range(9, 18))
        session_hours = sorted(random.sample(work_hours, min(session_count, len(work_hours))))
        
        for hour in session_hours:
            # Pick a script based on realistic usage patterns
            script_choice = random.choices(
                [s[0] for s in scripts],
                weights=[s[1] for s in scripts]
            )[0]
            
            # Get script parameters
            script_data = next(s for s in scripts if s[0] == script_choice)
            _, weight, avg_duration, avg_files = script_data
            
            # Add some randomness to duration and file count
            duration = max(15, int(random.gauss(avg_duration, avg_duration * 0.3)))
            files = max(10, int(random.gauss(avg_files, avg_files * 0.4)))
            
            # Generate session
            session = generate_realistic_session(script_choice, date, hour, duration, files)
            daily_sessions.append(session)
        
        # Save daily data
        daily_file = data_dir / f"daily_{date}.json"
        with open(daily_file, 'w') as f:
            json.dump(daily_sessions, f, indent=2)
            
        print(f"  📅 {date}: {len(daily_sessions)} sessions, "
              f"{sum(len(s['operations']) for s in daily_sessions)} operations")
    
    print(f"✅ Demo data generated in {data_dir}")
    return data_dir

def show_demo_analytics(data_dir):
    """Show what the analytics look like with demo data"""
    
    print("\n" + "="*80)
    print("🎯 DEMO ANALYTICS - This is what you'll see with real data!")
    print("="*80)
    
    # Create reporter with demo data
    reporter = TimerReporter()
    reporter.data_dir = data_dir
    
    # Show daily summary
    print("\n📊 SAMPLE DAILY REPORT:")
    print("-" * 50)
    reporter.print_daily_summary()
    
    # Show weekly summary  
    print("\n📈 SAMPLE WEEKLY REPORT:")
    print("-" * 50)
    reporter.print_cross_script_summary(7)
    
    # Show productivity metrics
    print("\n🚀 SAMPLE PRODUCTIVITY ANALYSIS:")
    print("-" * 50)
    
    # Import the advanced reporter
    sys.path.append(str(Path(__file__).parent))
    from utils.timer_report import AdvancedTimerReporter
    
    advanced_reporter = AdvancedTimerReporter()
    advanced_reporter.data_dir = data_dir
    advanced_reporter.productivity_metrics(14)
    
    # Show trend analysis
    print("\n📈 SAMPLE TREND ANALYSIS:")
    print("-" * 50)
    advanced_reporter.trend_analysis(14)

def show_integration_example():
    """Show how FileTracker and ActivityTimer data combine"""
    
    print("\n" + "="*80)
    print("🔗 FILETRACKER + ACTIVITY TIMER INTEGRATION EXAMPLE")
    print("="*80)
    
    print("""
🎯 SCENARIO: Processing 150 image triplets in web selector

FileTracker Records:
┌─────────────────────────────────────────────────────────────┐
│ [14:23:15] move: 89 files (XXX_CONTENT/ → Reviewed/)       │
│ [14:23:15] delete: 61 files (sent to trash)                │
│ [14:23:15] move: 89 YAML files (XXX_CONTENT/ → Reviewed/)  │
│ Total files handled: 300 (150 triplets)                    │
└─────────────────────────────────────────────────────────────┘

ActivityTimer Records:
┌─────────────────────────────────────────────────────────────┐
│ Session: 01_web_image_selector                              │
│ Duration: 12m 34s total, 10m 47s active (85.7% efficiency) │
│ Operations: select(89), delete(61), move(89)               │
│ Batch: "Triplet Selection Batch 15"                        │
│ User interactions: 456 (clicks: 298, keys: 158)            │
└─────────────────────────────────────────────────────────────┘

🎯 COMBINED INSIGHTS:
┌─────────────────────────────────────────────────────────────┐
│ • Processing Rate: 13.9 triplets/minute (above avg 11.2)   │
│ • Decision Speed: 4.3 seconds per triplet (excellent)      │
│ • Keep Rate: 59.3% (89/150 - typical range 55-65%)        │
│ • Efficiency: 85.7% (above daily avg 82.4%)               │
│ • Quality Score: HIGH (fast decisions + good efficiency)    │
│ • Recommendation: Maintain current batch size (150)        │
└─────────────────────────────────────────────────────────────┘

📊 PERFORMANCE COMPARISON:
                    This Session    Daily Avg    Weekly Avg
Files/Hour:         668            587          612
Efficiency:         85.7%          82.4%        81.9%
Keep Rate:          59.3%          61.2%        60.8%
Decision Time:      4.3s           5.1s         5.4s
    """)

def main():
    """Main demo function"""
    
    print("🎭 Activity Timer System - Data Demo")
    print("="*50)
    print("This will show you exactly what analytics you'll get!")
    print()
    
    # Generate demo data
    data_dir = generate_demo_data(14)
    
    # Show analytics
    show_demo_analytics(data_dir)
    
    # Show integration example
    show_integration_example()
    
    print("\n" + "="*80)
    print("🎉 This is what you'll see with REAL data from your workflow!")
    print("="*80)
    print("""
🚀 NEXT STEPS:
1. Start using 01_web_image_selector.py - timer is already integrated!
2. Run 'python scripts/util_timer_report.py' to see your real analytics
3. Use 'python scripts/util_timer_report.py --live' for live monitoring
4. Check daily reports with 'python scripts/util_timer_report.py --daily'

The timer will automatically:
✅ Track your active work time vs idle time
✅ Measure files processed per hour
✅ Calculate efficiency percentages  
✅ Identify your peak performance hours
✅ Show which tools work best for you
✅ Provide data-driven optimization insights

All data is stored locally and completely private! 🔒
    """)

if __name__ == "__main__":
    main()
