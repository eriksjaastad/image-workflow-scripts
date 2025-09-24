#!/usr/bin/env python3
"""
Productivity Dashboard - Web Interface
======================================
Flask-based web dashboard for visualizing workflow productivity data.

Features:
- Global and individual graph time controls
- Bar charts with historical average overlays
- Script update markers with hover descriptions
- Modular design for easy script additions
- Dark theme matching existing tools
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from data_engine import DashboardDataEngine

class ProductivityDashboard:
    def __init__(self, data_dir: str = ".."):
        self.data_engine = DashboardDataEngine(data_dir)
        # Set template folder to current directory
        self.app = Flask(__name__, template_folder='.')
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route("/")
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard_template.html')
        
        @self.app.route("/api/data/<time_slice>")
        def get_dashboard_data(time_slice):
            """API endpoint for dashboard data"""
            lookback_days = request.args.get('lookback_days', 30, type=int)
            
            try:
                data = self.data_engine.generate_dashboard_data(
                    time_slice=time_slice, 
                    lookback_days=lookback_days
                )
                
                # Transform data for Chart.js format
                chart_data = self.transform_for_charts(data)
                return jsonify(chart_data)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/scripts")
        def get_scripts():
            """Get list of discovered scripts"""
            scripts = self.data_engine.discover_scripts()
            return jsonify({"scripts": scripts})
        
        @self.app.route("/api/script_updates", methods=["GET", "POST"])
        def handle_script_updates():
            """Handle script update tracking"""
            if request.method == "POST":
                data = request.get_json()
                self.data_engine.add_script_update(
                    script=data.get('script'),
                    description=data.get('description'),
                    date=data.get('date')
                )
                return jsonify({"status": "success"})
            else:
                updates = self.data_engine.load_script_updates()
                return jsonify(updates.to_dict('records'))
    
    def transform_for_charts(self, data):
        """Transform raw data into Chart.js format"""
        chart_data = {
            "metadata": data["metadata"],
            "charts": {}
        }
        
        # Script name mapping for display
        script_display_names = {
            'image_version_selector': '01_web_image_selector',
            'character_sorter': '03_web_character_sorter',
            'batch_crop_tool': '04_batch_crop_tool'
        }
        
        # Transform file operations by script
        if data['file_operations_data'].get('by_script'):
            by_script_data = {}
            for record in data['file_operations_data']['by_script']:
                script = record['script']
                display_name = script_display_names.get(script, script)
                date = record['time_slice']
                count = record['file_count']
                
                if display_name not in by_script_data:
                    by_script_data[display_name] = {'dates': [], 'counts': []}
                
                by_script_data[display_name]['dates'].append(date)
                by_script_data[display_name]['counts'].append(count)
            
            chart_data['charts']['by_script'] = by_script_data
        
        # Transform file operations by type
        if data['file_operations_data'].get('by_operation'):
            by_operation_data = {}
            for record in data['file_operations_data']['by_operation']:
                operation = record['operation']
                date = record['time_slice']
                count = record['file_count']
                
                if operation not in by_operation_data:
                    by_operation_data[operation] = {'dates': [], 'counts': []}
                
                by_operation_data[operation]['dates'].append(date)
                by_operation_data[operation]['counts'].append(count)
            
            chart_data['charts']['by_operation'] = by_operation_data
        
        return chart_data
    
    def run(self, host="127.0.0.1", port=5001, debug=False):
        """Run the dashboard server"""
        import webbrowser
        import threading
        import time
        
        url = f"http://{host}:{port}"
        print(f"🚀 Productivity Dashboard starting at {url}")
        
        # Auto-open browser after a short delay (like your other scripts)
        def open_browser():
            time.sleep(1.5)  # Give server time to start
            try:
                webbrowser.open(url)
                print(f"🌐 Opening browser to {url}")
            except Exception as e:
                print(f"Could not auto-open browser: {e}")
        
        # Start browser opener in background thread
        if not debug:  # Don't auto-open in debug mode
            threading.Thread(target=open_browser, daemon=True).start()
        
        self.app.run(host=host, port=port, debug=debug)

# Dashboard HTML Template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Productivity Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            color-scheme: dark;
            --bg: #101014;
            --surface: #181821;
            --surface-alt: #1f1f2c;
            --accent: #4f9dff;
            --accent-soft: rgba(79, 157, 255, 0.2);
            --success: #51cf66;
            --danger: #ff6b6b;
            --warning: #ffd43b;
            --muted: #a0a3b1;
        }
        
        * { box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: white;
            min-height: 100vh;
        }
        
        .dashboard-header {
            background: var(--surface);
            padding: 1.5rem 2rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .dashboard-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent);
            margin: 0;
        }
        
        .global-controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .time-selector {
            background: var(--surface-alt);
            border: 1px solid rgba(255,255,255,0.1);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
        }
        
        .time-selector:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .dashboard-content {
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .chart-container {
            background: var(--surface);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .chart-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            margin: 0;
        }
        
        .chart-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .chart-time-btn {
            background: var(--surface-alt);
            border: 1px solid rgba(255,255,255,0.1);
            color: var(--muted);
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .chart-time-btn:hover {
            background: var(--accent-soft);
            color: white;
        }
        
        .chart-time-btn.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }
        
        .chart-canvas {
            height: 300px;
            margin-top: 1rem;
        }
        
        .loading {
            text-align: center;
            color: var(--muted);
            padding: 2rem;
        }
        
        .error {
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid var(--danger);
            color: var(--danger);
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }
        
        .stats-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: var(--surface);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: var(--muted);
            font-size: 0.9rem;
        }
        
        .update-marker {
            position: absolute;
            width: 2px;
            background: var(--warning);
            top: 0;
            bottom: 0;
            cursor: pointer;
        }
        
        .update-tooltip {
            position: absolute;
            background: var(--surface-alt);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            z-index: 1000;
            display: none;
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1 class="dashboard-title">📊 Productivity Dashboard</h1>
        <div class="global-controls">
            <label for="global-time">Global Time Scale:</label>
            <select id="global-time" class="time-selector">
                <option value="">Individual Controls</option>
                <option value="15min">15 Minutes</option>
                <option value="1H">1 Hour</option>
                <option value="D">Daily</option>
                <option value="W">Weekly</option>
                <option value="M">Monthly</option>
            </select>
            <select id="lookback-days" class="time-selector">
                <option value="7">Last 7 days</option>
                <option value="30" selected>Last 30 days</option>
                <option value="90">Last 90 days</option>
                <option value="365">Last year</option>
            </select>
        </div>
    </div>
    
    <div class="dashboard-content">
        <div class="stats-summary" id="stats-summary">
            <div class="stat-card">
                <div class="stat-value" id="total-active-time">--</div>
                <div class="stat-label">Total Active Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-files">--</div>
                <div class="stat-label">Files Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-efficiency">--</div>
                <div class="stat-label">Average Efficiency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="active-scripts">--</div>
                <div class="stat-label">Active Scripts</div>
            </div>
        </div>
        
        <div class="charts-grid" id="charts-container">
            <div class="loading">Loading dashboard data...</div>
        </div>
    </div>

    <script>
        // Dashboard state
        let dashboardData = null;
        let charts = {};
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            setupGlobalControls();
            loadDashboardData('D'); // Default to daily view
        });
        
        function setupGlobalControls() {
            const globalTimeSelect = document.getElementById('global-time');
            const lookbackSelect = document.getElementById('lookback-days');
            
            globalTimeSelect.addEventListener('change', function() {
                if (this.value) {
                    // Apply to all charts
                    updateAllCharts(this.value);
                }
            });
            
            lookbackSelect.addEventListener('change', function() {
                const currentTimeSlice = globalTimeSelect.value || 'D';
                loadDashboardData(currentTimeSlice, parseInt(this.value));
            });
        }
        
        async function loadDashboardData(timeSlice = 'D', lookbackDays = 30) {
            try {
                const response = await fetch(`/api/data/${timeSlice}?lookback_days=${lookbackDays}`);
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                dashboardData = data;
                updateStatsCards(data);
                renderCharts(data, timeSlice);
                
            } catch (error) {
                showError('Failed to load dashboard data: ' + error.message);
            }
        }
        
        function updateStatsCards(data) {
            // Calculate summary statistics
            let totalActiveTime = 0;
            let totalFiles = 0;
            let totalEfficiency = 0;
            let efficiencyCount = 0;
            
            if (data.activity_data.active_time) {
                data.activity_data.active_time.forEach(record => {
                    totalActiveTime += record.active_time || 0;
                });
            }
            
            if (data.activity_data.files_processed) {
                data.activity_data.files_processed.forEach(record => {
                    totalFiles += record.files_processed || 0;
                });
            }
            
            if (data.activity_data.efficiency) {
                data.activity_data.efficiency.forEach(record => {
                    if (record.efficiency > 0) {
                        totalEfficiency += record.efficiency;
                        efficiencyCount++;
                    }
                });
            }
            
            // Update cards
            document.getElementById('total-active-time').textContent = 
                Math.round(totalActiveTime / 60) + 'm';
            document.getElementById('total-files').textContent = totalFiles;
            document.getElementById('avg-efficiency').textContent = 
                efficiencyCount > 0 ? Math.round(totalEfficiency / efficiencyCount) + '%' : '--';
            document.getElementById('active-scripts').textContent = 
                data.metadata.scripts_found.length;
        }
        
        function renderCharts(data, timeSlice) {
            const container = document.getElementById('charts-container');
            container.innerHTML = '';
            
            // Render different chart types
            if (data.activity_data.active_time && data.activity_data.active_time.length > 0) {
                renderBarChart(container, 'Active Time by Script', data.activity_data.active_time, 
                             'active_time', timeSlice, data.historical_averages.active_time);
            }
            
            if (data.activity_data.files_processed && data.activity_data.files_processed.length > 0) {
                renderBarChart(container, 'Files Processed by Script', data.activity_data.files_processed, 
                             'files_processed', timeSlice, data.historical_averages.files_processed);
            }
            
            if (data.file_operations_data.deletions && data.file_operations_data.deletions.length > 0) {
                renderBarChart(container, 'Files Deleted by Script', data.file_operations_data.deletions, 
                             'file_count', timeSlice);
            }
            
            if (data.file_operations_data.by_operation && data.file_operations_data.by_operation.length > 0) {
                renderBarChart(container, 'Operations by Type', data.file_operations_data.by_operation, 
                             'file_count', timeSlice, null, 'operation');
            }
        }
        
        function renderBarChart(container, title, data, valueField, timeSlice, averageData = null, groupField = 'script') {
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            
            const chartId = 'chart-' + title.toLowerCase().replace(/[^a-z0-9]/g, '-');
            
            chartDiv.innerHTML = `
                <div class="chart-header">
                    <h3 class="chart-title">${title}</h3>
                    <div class="chart-controls">
                        <button class="chart-time-btn ${timeSlice === '15min' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', '15min')">15m</button>
                        <button class="chart-time-btn ${timeSlice === '1H' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', '1H')">1h</button>
                        <button class="chart-time-btn ${timeSlice === 'D' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', 'D')">Daily</button>
                        <button class="chart-time-btn ${timeSlice === 'W' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', 'W')">Weekly</button>
                        <button class="chart-time-btn ${timeSlice === 'M' ? 'active' : ''}" 
                                onclick="updateChart('${chartId}', 'M')">Monthly</button>
                    </div>
                </div>
                <div class="chart-canvas">
                    <canvas id="${chartId}"></canvas>
                </div>
            `;
            
            container.appendChild(chartDiv);
            
            // Create Chart.js chart
            const ctx = document.getElementById(chartId).getContext('2d');
            
            // Process data for Chart.js
            const processedData = processChartData(data, valueField, groupField);
            
            const chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: processedData.labels,
                    datasets: [{
                        label: title,
                        data: processedData.values,
                        backgroundColor: 'rgba(79, 157, 255, 0.8)',
                        borderColor: 'rgba(79, 157, 255, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: 'white'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: '#a0a3b1'
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            }
                        },
                        y: {
                            ticks: {
                                color: '#a0a3b1'
                            },
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            }
                        }
                    }
                }
            });
            
            charts[chartId] = {
                chart: chart,
                title: title,
                data: data,
                valueField: valueField,
                groupField: groupField,
                averageData: averageData
            };
        }
        
        function processChartData(data, valueField, groupField) {
            const grouped = {};
            
            data.forEach(record => {
                const key = record[groupField] || 'unknown';
                if (!grouped[key]) {
                    grouped[key] = 0;
                }
                grouped[key] += record[valueField] || 0;
            });
            
            return {
                labels: Object.keys(grouped),
                values: Object.values(grouped)
            };
        }
        
        function updateChart(chartId, timeSlice) {
            // Update individual chart time slice
            const lookbackDays = parseInt(document.getElementById('lookback-days').value);
            loadDashboardData(timeSlice, lookbackDays);
        }
        
        function updateAllCharts(timeSlice) {
            // Update all charts to the same time slice
            const lookbackDays = parseInt(document.getElementById('lookback-days').value);
            loadDashboardData(timeSlice, lookbackDays);
        }
        
        function showError(message) {
            const container = document.getElementById('charts-container');
            container.innerHTML = `<div class="error">Error: ${message}</div>`;
        }
    </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Productivity Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", default=5001, type=int, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--data-dir", default="scripts", help="Data directory path")
    
    args = parser.parse_args()
    
    dashboard = ProductivityDashboard(args.data_dir)
    dashboard.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
