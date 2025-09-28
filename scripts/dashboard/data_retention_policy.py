#!/usr/bin/env python3
"""
📦 Smart Data Retention Policy
Keeps dashboard data efficient and prevents storage explosion
"""

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import gzip
import shutil

class DataRetentionManager:
    def __init__(self, data_dir: str = ".."):
        self.data_dir = Path(data_dir)
        self.timer_data_dir = self.data_dir / "data" / "timer_data"
        self.file_ops_dir = self.data_dir / "data" / "file_operations_logs"
        self.archive_dir = self.data_dir / "data" / "dashboard_archives"
        
        # Retention policies (days) - Realistic for contract work
        self.DETAILED_RETENTION = 30    # Keep full detail for 30 days
        self.SUMMARY_RETENTION = 365    # Keep daily summaries for 1 year
        self.ARCHIVE_RETENTION = 365    # Keep archives for 1 year max (AI might take over by then! 🤖)
        
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage usage statistics"""
        stats = {
            "timer_data": {"size_mb": 0, "files": 0, "oldest": None, "newest": None},
            "file_ops": {"size_mb": 0, "files": 0, "oldest": None, "newest": None},
            "archives": {"size_mb": 0, "files": 0, "oldest": None, "newest": None},
            "total_size_mb": 0
        }
        
        # Check timer data
        if self.timer_data_dir.exists():
            timer_files = list(self.timer_data_dir.glob("*.json"))
            if timer_files:
                stats["timer_data"]["files"] = len(timer_files)
                total_size = sum(f.stat().st_size for f in timer_files)
                stats["timer_data"]["size_mb"] = round(total_size / 1024 / 1024, 2)
                
                dates = [f.stem.split('_')[1] for f in timer_files if '_' in f.stem]
                if dates:
                    stats["timer_data"]["oldest"] = min(dates)
                    stats["timer_data"]["newest"] = max(dates)
        
        # Check file operations
        if self.file_ops_dir.exists():
            ops_files = list(self.file_ops_dir.glob("*.log"))
            if ops_files:
                stats["file_ops"]["files"] = len(ops_files)
                total_size = sum(f.stat().st_size for f in ops_files)
                stats["file_ops"]["size_mb"] = round(total_size / 1024 / 1024, 2)
        
        # Check archives
        if self.archive_dir.exists():
            archive_files = list(self.archive_dir.glob("*.gz"))
            if archive_files:
                stats["archives"]["files"] = len(archive_files)
                total_size = sum(f.stat().st_size for f in archive_files)
                stats["archives"]["size_mb"] = round(total_size / 1024 / 1024, 2)
        
        stats["total_size_mb"] = (
            stats["timer_data"]["size_mb"] + 
            stats["file_ops"]["size_mb"] + 
            stats["archives"]["size_mb"]
        )
        
        return stats
    
    def create_daily_summary(self, date_str: str) -> Dict[str, Any]:
        """Create a compact daily summary from detailed data"""
        summary = {
            "date": date_str,
            "scripts": {},
            "totals": {
                "active_time": 0,
                "total_time": 0,
                "files_processed": 0,
                "operations": {}
            }
        }
        
        # Process timer data
        timer_file = self.timer_data_dir / f"daily_{date_str}.json"
        if timer_file.exists():
            try:
                with open(timer_file, 'r') as f:
                    data = json.load(f)
                
                for script, sessions in data.get("scripts", {}).items():
                    script_summary = {
                        "sessions": len(sessions),
                        "active_time": sum(s.get("active_time", 0) for s in sessions),
                        "total_time": sum(s.get("total_time", 0) for s in sessions),
                        "files_processed": sum(s.get("files_processed", 0) for s in sessions),
                        "operations": {}
                    }
                    
                    # Aggregate operations
                    for session in sessions:
                        for op, count in session.get("operations", {}).items():
                            script_summary["operations"][op] = script_summary["operations"].get(op, 0) + count
                            summary["totals"]["operations"][op] = summary["totals"]["operations"].get(op, 0) + count
                    
                    summary["scripts"][script] = script_summary
                    summary["totals"]["active_time"] += script_summary["active_time"]
                    summary["totals"]["total_time"] += script_summary["total_time"]
                    summary["totals"]["files_processed"] += script_summary["files_processed"]
                    
            except Exception as e:
                print(f"Warning: Could not process {timer_file}: {e}")
        
        return summary
    
    def compress_old_data(self, cutoff_days: int = 30) -> List[str]:
        """Compress data older than cutoff_days"""
        cutoff_date = datetime.now() - timedelta(days=cutoff_days)
        cutoff_str = cutoff_date.strftime("%Y%m%d")
        
        self.archive_dir.mkdir(exist_ok=True)
        compressed_files = []
        
        # Compress old timer data
        if self.timer_data_dir.exists():
            for timer_file in self.timer_data_dir.glob("daily_*.json"):
                date_part = timer_file.stem.split('_')[1]
                if date_part < cutoff_str:
                    # Create summary before compression
                    summary = self.create_daily_summary(date_part)
                    summary_file = self.archive_dir / f"summary_{date_part}.json"
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                    
                    # Compress original file
                    archive_file = self.archive_dir / f"{timer_file.name}.gz"
                    with open(timer_file, 'rb') as f_in:
                        with gzip.open(archive_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    timer_file.unlink()  # Delete original
                    compressed_files.append(str(archive_file))
        
        return compressed_files
    
    def cleanup_old_archives(self, max_age_days: int = 1095) -> List[str]:
        """Remove archives older than max_age_days"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_files = []
        
        if self.archive_dir.exists():
            for archive_file in self.archive_dir.glob("*.gz"):
                if archive_file.stat().st_mtime < cutoff_date.timestamp():
                    archive_file.unlink()
                    removed_files.append(str(archive_file))
        
        return removed_files
    
    def optimize_file_operations_log(self, max_size_mb: int = 50) -> bool:
        """Rotate file operations log if it gets too large"""
        log_file = self.file_ops_dir / "file_operations.log"
        
        if not log_file.exists():
            return False
        
        size_mb = log_file.stat().st_size / 1024 / 1024
        
        if size_mb > max_size_mb:
            # Rotate log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_file = self.file_ops_dir / f"file_operations_{timestamp}.log"
            
            # Move current log to rotated name
            shutil.move(log_file, rotated_file)
            
            # Compress the rotated log
            self.archive_dir.mkdir(exist_ok=True)
            compressed_file = self.archive_dir / f"{rotated_file.name}.gz"
            with open(rotated_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            rotated_file.unlink()  # Remove uncompressed rotated file
            
            print(f"📦 Rotated and compressed log: {compressed_file}")
            return True
        
        return False
    
    def run_maintenance(self) -> Dict[str, Any]:
        """Run complete data maintenance cycle"""
        print("🧹 Running data maintenance...")
        
        results = {
            "before_stats": self.get_storage_stats(),
            "compressed_files": [],
            "removed_files": [],
            "log_rotated": False
        }
        
        # Compress old detailed data (keep summaries)
        results["compressed_files"] = self.compress_old_data(self.DETAILED_RETENTION)
        
        # Remove very old archives
        results["removed_files"] = self.cleanup_old_archives(self.ARCHIVE_RETENTION)
        
        # Rotate large log files
        results["log_rotated"] = self.optimize_file_operations_log()
        
        results["after_stats"] = self.get_storage_stats()
        
        # Calculate space saved
        space_saved = results["before_stats"]["total_size_mb"] - results["after_stats"]["total_size_mb"]
        results["space_saved_mb"] = round(space_saved, 2)
        
        print(f"✅ Maintenance complete! Saved {space_saved:.2f} MB")
        return results

def main():
    """Demo the data retention system"""
    print("📦 Data Retention Policy Demo")
    print("=" * 50)
    
    manager = DataRetentionManager()
    stats = manager.get_storage_stats()
    
    print("📊 Current Storage Usage:")
    print(f"  Timer Data: {stats['timer_data']['size_mb']:.2f} MB ({stats['timer_data']['files']} files)")
    print(f"  File Ops:   {stats['file_ops']['size_mb']:.2f} MB ({stats['file_ops']['files']} files)")
    print(f"  Archives:   {stats['archives']['size_mb']:.2f} MB ({stats['archives']['files']} files)")
    print(f"  Total:      {stats['total_size_mb']:.2f} MB")
    
    print(f"\n🔄 Retention Policy:")
    print(f"  Detailed data: {manager.DETAILED_RETENTION} days")
    print(f"  Daily summaries: {manager.SUMMARY_RETENTION} days") 
    print(f"  Compressed archives: {manager.ARCHIVE_RETENTION} days")
    
    print(f"\n💡 This prevents storage explosion while keeping useful data!")

if __name__ == "__main__":
    main()
