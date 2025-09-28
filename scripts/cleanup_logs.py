#!/usr/bin/env python3
"""
🧹 Log Cleanup Tool - Optimized for Erik's File Structure
=========================================================
Cleans up large log files while preserving data needed for dashboard.

Based on Erik's requirements:
- Keep 2 days of detailed file operations logs (for tracking lost files)
- Archive older logs for dashboard historical data
- Compress old data to save space
"""

import os
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json

class LogCleanupManager:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.file_ops_dir = self.data_dir / "file_operations_logs"
        self.timer_data_dir = self.data_dir / "timer_data"
        self.archive_dir = self.data_dir / "log_archives"
        
        # Retention policies
        self.DETAILED_LOG_DAYS = 2      # Keep 2 days of detailed file ops (for lost file tracking)
        self.DASHBOARD_ARCHIVE_DAYS = 365  # Keep dashboard data for 1 year
        
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        if file_path.exists():
            return file_path.stat().st_size / 1024 / 1024
        return 0.0
    
    def analyze_current_usage(self):
        """Analyze current log file usage"""
        print("📊 Current Log File Analysis")
        print("=" * 50)
        
        # File operations logs
        if self.file_ops_dir.exists():
            log_files = list(self.file_ops_dir.glob("*.log"))
            total_size = sum(self.get_file_size_mb(f) for f in log_files)
            
            print(f"📁 File Operations Logs: {len(log_files)} files, {total_size:.1f} MB")
            
            # Show largest files
            log_files.sort(key=lambda f: f.stat().st_size, reverse=True)
            for i, log_file in enumerate(log_files[:5]):
                size_mb = self.get_file_size_mb(log_file)
                age_days = (datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)).days
                print(f"  {i+1}. {log_file.name}: {size_mb:.1f} MB ({age_days} days old)")
        
        # Timer data
        if self.timer_data_dir.exists():
            timer_files = list(self.timer_data_dir.glob("*.json"))
            total_size = sum(self.get_file_size_mb(f) for f in timer_files)
            print(f"📁 Timer Data: {len(timer_files)} files, {total_size:.1f} MB")
        
        print()
    
    def cleanup_file_operations_logs(self):
        """Clean up file operations logs - keep only recent ones"""
        if not self.file_ops_dir.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=self.DETAILED_LOG_DAYS)
        
        # Create archive directory
        self.archive_dir.mkdir(exist_ok=True)
        
        cleaned_files = []
        preserved_files = []
        
        for log_file in self.file_ops_dir.glob("*.log"):
            file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            if file_date < cutoff_date:
                # Archive old log file
                archive_name = f"{log_file.stem}_archived.log.gz"
                archive_path = self.archive_dir / archive_name
                
                # Compress and archive
                with open(log_file, 'rb') as f_in:
                    with gzip.open(archive_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original
                original_size = self.get_file_size_mb(log_file)
                log_file.unlink()
                
                cleaned_files.append({
                    'original': str(log_file),
                    'archive': str(archive_path),
                    'original_size_mb': original_size,
                    'compressed_size_mb': self.get_file_size_mb(archive_path)
                })
            else:
                preserved_files.append(str(log_file))
        
        return cleaned_files, preserved_files
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        print("🧹 Starting Log Cleanup Process")
        print("=" * 50)
        
        # Show current state
        self.analyze_current_usage()
        
        # Clean up file operations logs
        print("🗂️ Cleaning up file operations logs...")
        cleaned_files, preserved_files = self.cleanup_file_operations_logs()
        
        if cleaned_files:
            total_original = sum(f['original_size_mb'] for f in cleaned_files)
            total_compressed = sum(f['compressed_size_mb'] for f in cleaned_files)
            space_saved = total_original - total_compressed
            
            print(f"✅ Archived {len(cleaned_files)} old log files")
            print(f"💾 Space saved: {space_saved:.1f} MB ({total_original:.1f} MB → {total_compressed:.1f} MB)")
            print(f"📦 Archived files stored in: {self.archive_dir}")
        else:
            print("ℹ️ No old log files found to clean up")
        
        if preserved_files:
            print(f"🔒 Preserved {len(preserved_files)} recent log files (last {self.DETAILED_LOG_DAYS} days)")
        
        print(f"\n📋 Summary:")
        print(f"  • Recent logs kept: {len(preserved_files)} files (for lost file tracking)")
        print(f"  • Old logs archived: {len(cleaned_files)} files (for dashboard history)")
        print(f"  • Archive location: {self.archive_dir}")
        
        return {
            'cleaned_files': cleaned_files,
            'preserved_files': preserved_files,
            'archive_dir': str(self.archive_dir)
        }

def main():
    """Run the log cleanup"""
    manager = LogCleanupManager()
    results = manager.run_cleanup()
    
    print(f"\n🎉 Log cleanup complete!")
    
    if results['cleaned_files']:
        print(f"Run this command again in the future to maintain clean logs.")
    
    return results

if __name__ == "__main__":
    main()
