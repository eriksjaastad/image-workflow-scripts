"""
Timesheet Parser - Reads and aggregates timesheet CSV data for dashboard
"""
import csv
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class TimesheetParser:
    """Parse timesheet CSV and aggregate data by project"""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
    
    def parse(self) -> Dict[str, Any]:
        """
        Parse timesheet CSV and return aggregated project data
        
        Returns:
            {
                "projects": [
                    {
                        "name": "mojo-1",
                        "total_hours": 70,
                        "starting_images": 19183,
                        "final_images": 6453,
                        "images_per_hour": 274.0,
                        "hours_per_image": 0.0036,
                        "start_date": "10/1/2025",
                        "last_date": "10/11/2025"
                    },
                    ...
                ],
                "totals": {
                    "total_hours": 231,
                    "total_projects": 15
                }
            }
        """
        if not self.csv_path.exists():
            return {"projects": [], "totals": {"total_hours": 0, "total_projects": 0}}
        
        projects = {}
        current_project = None
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                date = row.get('Date', '').strip()
                project_name = row.get('Project', '').strip()
                hours_str = row.get('Hours', '').strip()
                starting_images_str = row.get('Starting Images', '').strip()
                final_images_str = row.get('Final Images', '').strip()
                
                # Skip summary/total rows (rows with no date and no project name)
                if not date and not project_name:
                    continue
                
                # Parse hours for this row
                try:
                    hours = float(hours_str) if hours_str else 0
                except ValueError:
                    hours = 0
                
                # If there's a project name, this starts a new project
                if project_name:
                    current_project = project_name
                    
                    if current_project not in projects:
                        projects[current_project] = {
                            'name': current_project,
                            'total_hours': 0,
                            'starting_images': 0,
                            'final_images': 0,
                            'start_date': date,
                            'last_date': date,
                            'dates': []
                        }
                    
                    # Parse starting images (only set once per project, usually first occurrence)
                    if starting_images_str and not projects[current_project]['starting_images']:
                        try:
                            projects[current_project]['starting_images'] = int(starting_images_str)
                        except ValueError:
                            pass
                    
                    # Parse final images (can be updated as project progresses)
                    if final_images_str:
                        try:
                            projects[current_project]['final_images'] = int(final_images_str)
                        except ValueError:
                            pass
                
                # Add hours to current project
                if current_project and hours > 0:
                    projects[current_project]['total_hours'] += hours
                    if date:
                        projects[current_project]['last_date'] = date
                        projects[current_project]['dates'].append(date)
        
        # Calculate efficiency metrics
        project_list = []
        total_hours = 0
        
        for proj_name, proj_data in projects.items():
            total_hours_proj = proj_data['total_hours']
            starting_images = proj_data['starting_images']
            
            # Calculate efficiency
            images_per_hour = round(starting_images / total_hours_proj, 1) if total_hours_proj > 0 and starting_images > 0 else 0
            hours_per_image = round(total_hours_proj / starting_images, 4) if starting_images > 0 else 0
            
            project_list.append({
                'name': proj_name,
                'total_hours': total_hours_proj,
                'starting_images': starting_images,
                'final_images': proj_data['final_images'],
                'images_per_hour': images_per_hour,
                'hours_per_image': hours_per_image,
                'start_date': proj_data['start_date'],
                'last_date': proj_data['last_date'],
                'date_count': len(proj_data['dates'])
            })
            
            total_hours += total_hours_proj
        
        # Sort by start date (chronological)
        project_list.sort(key=lambda x: self._parse_date(x['start_date']))
        
        return {
            'projects': project_list,
            'totals': {
                'total_hours': total_hours,
                'total_projects': len(project_list)
            }
        }
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime for sorting"""
        try:
            return datetime.strptime(date_str, '%m/%d/%Y')
        except (ValueError, TypeError):
            return datetime.min


if __name__ == '__main__':
    # Test the parser
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "timesheet.csv"
    
    parser = TimesheetParser(csv_path)
    data = parser.parse()
    
    print("=== Timesheet Data ===")
    print(f"Total Hours: {data['totals']['total_hours']}")
    print(f"Total Projects: {data['totals']['total_projects']}")
    print("\n=== Projects ===")
    
    for proj in data['projects']:
        print(f"\n{proj['name']}:")
        print(f"  Hours: {proj['total_hours']}")
        print(f"  Starting Images: {proj['starting_images']}")
        print(f"  Final Images: {proj['final_images']}")
        print(f"  Efficiency: {proj['images_per_hour']} img/hr ({proj['hours_per_image']} hr/img)")
        print(f"  Dates: {proj['start_date']} to {proj['last_date']} ({proj['date_count']} days)")

