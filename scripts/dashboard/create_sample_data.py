#!/usr/bin/env python3
"""
🎨 Create Sample Dashboard Data
Generate realistic sample data to demonstrate the dashboard
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import random

def create_sample_timer_data():
    """Create sample timer data files"""
    timer_dir = Path("../timer_data")
    timer_dir.mkdir(exist_ok=True)
    
    scripts = ['01_web_image_selector', '03_web_character_sorter', '04_batch_crop_tool']
    
    # Generate data for last 14 days
    for i in range(14):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        
        daily_data = {
            "date": date_str,
            "scripts": {}
        }
        
        # Add random sessions for each script
        for script in scripts:
            if random.random() > 0.3:  # 70% chance of activity
                sessions = []
                for session_num in range(random.randint(1, 3)):
                    session = {
                        "session_id": f"{date_str}_{script}_{session_num:02d}",
                        "script": script,
                        "start_time": date.timestamp() + random.randint(0, 86400),
                        "end_time": None,
                        "active_time": random.randint(300, 7200),  # 5min to 2hrs
                        "total_time": random.randint(400, 8000),
                        "files_processed": random.randint(50, 5000),
                        "operations": {
                            "delete": random.randint(10, 1000),
                            "crop": random.randint(5, 500),
                            "sort": random.randint(20, 2000),
                            "move": random.randint(15, 800)
                        }
                    }
                    sessions.append(session)
                
                daily_data["scripts"][script] = sessions
        
        # Save daily file
        daily_file = timer_dir / f"daily_{date_str}.json"
        with open(daily_file, 'w') as f:
            json.dump(daily_data, f, indent=2)
    
    print(f"✅ Created {timer_dir} with 14 days of sample timer data")

def create_sample_file_operations():
    """Create sample file operations log"""
    ops_dir = Path("../file_operations_logs")
    ops_dir.mkdir(exist_ok=True)
    
    log_file = ops_dir / "file_operations.log"
    
    scripts = {
        'image_version_selector': '01_web_image_selector',
        'character_sorter': '03_web_character_sorter', 
        'batch_crop_tool': '04_batch_crop_tool'
    }
    
    operations = ['delete', 'crop', 'sort', 'move', 'send_to_trash']
    
    log_entries = []
    
    # Generate entries for last 14 days
    for i in range(14):
        date = datetime.now() - timedelta(days=i)
        
        for script_key, script_display in scripts.items():
            if random.random() > 0.2:  # 80% chance of activity
                for _ in range(random.randint(1, 5)):  # 1-5 operations per day
                    timestamp = date + timedelta(
                        hours=random.randint(8, 20),
                        minutes=random.randint(0, 59)
                    )
                    
                    operation = random.choice(operations)
                    file_count = random.randint(1, 1000)
                    
                    # Format: timestamp,script,operation,file_count,session_id,batch_id
                    entry = f"{timestamp.isoformat()},{script_key},{operation},{file_count},session_{date.strftime('%Y%m%d')},batch_001"
                    log_entries.append(entry)
    
    # Sort by timestamp
    log_entries.sort()
    
    # Write log file
    with open(log_file, 'w') as f:
        f.write("timestamp,script,operation,file_count,session_id,batch_id\n")
        for entry in log_entries:
            f.write(entry + "\n")
    
    print(f"✅ Created {log_file} with {len(log_entries)} sample operations")

def create_script_updates():
    """Create sample script updates CSV"""
    updates_file = Path("script_updates.csv")
    
    updates = [
        "2025-09-15,01_web_image_selector,Added activity timer integration",
        "2025-09-16,03_web_character_sorter,Enhanced batch processing UI",
        "2025-09-18,04_batch_crop_tool,Improved crop handle sizing",
        "2025-09-20,01_web_image_selector,Fixed timer display positioning",
        "2025-09-22,03_web_character_sorter,Added keyboard shortcuts",
        "2025-09-23,04_batch_crop_tool,Optimized image loading performance"
    ]
    
    with open(updates_file, 'w') as f:
        f.write("date,script,description\n")
        for update in updates:
            f.write(update + "\n")
    
    print(f"✅ Created {updates_file} with sample script updates")

def main():
    print("🎨 Creating Sample Dashboard Data...")
    print("=" * 50)
    
    create_sample_timer_data()
    create_sample_file_operations()
    create_script_updates()
    
    print("\n🚀 Sample data created successfully!")
    print("\n📊 You can now test the dashboard with:")
    print("   python test_dashboard_data.py")
    print("   python run_dashboard.py  # (requires: pip install flask)")
    
    print(f"\n📁 Files created:")
    print(f"   📂 ../timer_data/ (14 daily files)")
    print(f"   📂 ../file_operations_logs/file_operations.log")
    print(f"   📄 script_updates.csv")

if __name__ == "__main__":
    main()
