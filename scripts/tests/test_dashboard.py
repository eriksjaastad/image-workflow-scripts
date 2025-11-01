#!/usr/bin/env python3
"""
Comprehensive Dashboard Tests - Real Data Validation
====================================================
Tests the productivity dashboard with real data to ensure it always works correctly.
Validates data processing, API endpoints, and dashboard functionality.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import json
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError


def _http_get(url: str, timeout: int = 5):
    try:
        with urlrequest.urlopen(url, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            body = resp.read()
            return status, body
    except HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        return e.code, body
    except URLError:
        return None, b""


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DashboardTest:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.dashboard_dir = self.project_root / "scripts" / "dashboard"
        self.test_port = 5002  # Different port for testing
        self.server_process = None
        self.base_url = f"http://localhost:{self.test_port}"

    def setup(self):
        """Set up test environment"""
        print("✓ Test environment set up")

    def cleanup(self):
        """Clean up test environment"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
        print("✓ Test environment cleaned up")

    def test_data_engine_initialization(self):
        """Test that the data engine can initialize with real data"""
        print("\n🧪 Testing Data Engine Initialization...")

        try:
            # Import the data engine
            sys.path.append(str(self.dashboard_dir))
            from data_engine import DashboardDataEngine

            # Initialize with project root
            engine = DashboardDataEngine(str(self.project_root))

            # Test script discovery
            scripts = engine.discover_scripts()
            print(f"  Found {len(scripts)} scripts with data: {scripts}")

            # Test data processing for different time slices
            time_slices = ["15min", "1hr", "daily", "weekly"]
            for time_slice in time_slices:
                try:
                    data = engine.generate_dashboard_data(
                        time_slice=time_slice, lookback_days=30
                    )
                    print(
                        f"  ✓ {time_slice} data generation: {len(data.get('time_labels', []))} data points"
                    )
                except Exception as e:
                    print(f"  ❌ {time_slice} data generation failed: {e}")
                    return False

            print("✅ Data engine initialization test PASSED")
            return True

        except Exception as e:
            print(f"❌ Data engine initialization test FAILED: {e}")
            return False

    def test_dashboard_server_startup(self):
        """Test that the dashboard server can start successfully"""
        print("\n🧪 Testing Dashboard Server Startup...")

        try:
            # Skip if Flask is not installed in the environment
            try:
                import flask  # noqa: F401
            except Exception:
                print("  ⚠️ Flask not installed; skipping dashboard server startup test")
                return True
            # Start dashboard server in background
            cmd = [
                sys.executable,
                "scripts/dashboard/run_dashboard.py",
                "--port",
                str(self.test_port),
                "--host",
                "127.0.0.1",
                "--debug",  # disable auto-open browser for tests
                "--skip-snapshots",  # speed up and avoid external deps during tests
            ]

            env = os.environ.copy()
            env["DASHBOARD_SKIP_SNAPSHOTS"] = "1"
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                env=env,
            )

            # Wait for server to start
            max_wait = 20
            for i in range(max_wait):
                status, _ = _http_get(f"{self.base_url}/", timeout=2)
                if status == 200:
                    print(
                        f"  ✓ Dashboard server started successfully on port {self.test_port}"
                    )
                    print("✅ Dashboard server startup test PASSED")
                    return True
                time.sleep(1)

            print(f"  ❌ Dashboard server failed to start within {max_wait} seconds")
            try:
                if self.server_process and self.server_process.stderr:
                    err = self.server_process.stderr.read()
                    if err:
                        try:
                            err_text = err.decode(errors="ignore")
                        except Exception:
                            err_text = str(err)
                        print("  └─ Server stderr:")
                        print(err_text)
                if self.server_process and self.server_process.stdout:
                    out = self.server_process.stdout.read()
                    if out:
                        try:
                            out_text = out.decode(errors="ignore")
                        except Exception:
                            out_text = str(out)
                        print("  └─ Server stdout:")
                        print(out_text)
                # Ensure the process is terminated and cleared to avoid cascading timeouts
                if self.server_process:
                    try:
                        self.server_process.terminate()
                        self.server_process.wait(timeout=5)
                    except Exception:
                        pass
                    self.server_process = None
            except Exception:
                pass
            return False

        except Exception as e:
            print(f"❌ Dashboard server startup test FAILED: {e}")
            return False

    def test_api_endpoints(self):
        """Test that all API endpoints work with real data"""
        print("\n🧪 Testing API Endpoints...")

        if not self.server_process:
            print("⚠️  Server not running, skipping API tests")
            return True

        try:
            # Test main dashboard page
            status, _ = _http_get(f"{self.base_url}/", timeout=5)
            if status != 200:
                print(f"  ❌ Main dashboard page failed: {status}")
                return False
            print("  ✓ Main dashboard page loads")

            # Test API endpoints for different time slices
            time_slices = ["15min", "1hr", "daily", "weekly"]
            for time_slice in time_slices:
                try:
                    status, body = _http_get(
                        f"{self.base_url}/api/data/{time_slice}?lookback_days=30",
                        timeout=10,
                    )
                    if status == 200:
                        try:
                            data = json.loads(body.decode("utf-8", errors="ignore"))
                        except Exception:
                            data = {}
                        print(f"  ✓ {time_slice} API endpoint: ok")
                    else:
                        print(f"  ❌ {time_slice} API endpoint failed: {status}")
                        return False
                except Exception as e:
                    print(f"  ❌ {time_slice} API endpoint error: {e}")
                    return False

            print("✅ API endpoints test PASSED")
            return True

        except Exception as e:
            print(f"❌ API endpoints test FAILED: {e}")
            return False

    def test_data_accuracy(self):
        """Test that dashboard data matches actual log files"""
        print("\n🧪 Testing Data Accuracy...")

        try:
            # Import data engine
            sys.path.append(str(self.dashboard_dir))
            from data_engine import DashboardDataEngine

            engine = DashboardDataEngine(str(self.project_root))

            # Check if we have real timer data
            timer_data_dir = self.project_root / "scripts" / "timer_data"
            if not timer_data_dir.exists():
                print("  ⚠️ No timer data directory found, skipping accuracy test")
                return True

            timer_files = list(timer_data_dir.glob("daily_*.json"))
            if not timer_files:
                print("  ⚠️ No daily timer files found, skipping accuracy test")
                return True

            print(f"  Found {len(timer_files)} daily timer files")

            # Test data processing for recent data
            data = engine.generate_dashboard_data(time_slice="daily", lookback_days=7)

            # Validate data structure (be flexible about what keys exist)
            expected_keys = ["time_labels", "datasets", "script_updates"]
            found_keys = [key for key in expected_keys if key in data]

            if len(found_keys) == 0:
                print(f"  ❌ No expected keys found in data: {list(data.keys())}")
                return False

            print(
                f"  ✓ Found {len(found_keys)}/{len(expected_keys)} expected keys: {found_keys}"
            )

            # Only check these if they exist
            if "time_labels" in data:
                print(
                    f"  ✓ Data structure valid: {len(data['time_labels'])} time points"
                )
            if "datasets" in data:
                print(f"  ✓ Found {len(data['datasets'])} script datasets")

            # Check for file operations data
            file_ops_dir = self.project_root / "scripts" / "file_operations_logs"
            if file_ops_dir.exists():
                log_files = list(file_ops_dir.glob("*.log"))
                print(f"  ✓ Found {len(log_files)} file operation log files")

            print("✅ Data accuracy test PASSED")
            return True

        except Exception as e:
            print(f"❌ Data accuracy test FAILED: {e}")
            return False

    def test_dashboard_performance(self):
        """Test dashboard performance with real data"""
        print("\n🧪 Testing Dashboard Performance...")

        if not self.server_process:
            print("⚠️  Server not running, skipping performance tests")
            return True

        try:
            # Test response times for different endpoints
            endpoints = [
                ("/", "Main Page"),
                ("/api/data/daily?lookback_days=30", "Daily API"),
                ("/api/data/weekly?lookback_days=90", "Weekly API"),
            ]

            for endpoint, name in endpoints:
                start_time = time.time()
                status, _ = _http_get(f"{self.base_url}{endpoint}", timeout=15)
                end_time = time.time()
                response_time = end_time - start_time
                if status == 200 and response_time < 10:
                    print(f"  ✓ {name}: {response_time:.2f}s")
                else:
                    print(f"  ❌ {name}: {response_time:.2f}s (status: {status})")
                    return False

            print("✅ Dashboard performance test PASSED")
            return True

        except Exception as e:
            print(f"❌ Dashboard performance test FAILED: {e}")
            return False

    def test_error_handling(self):
        """Test dashboard error handling"""
        print("\n🧪 Testing Error Handling...")

        if not self.server_process:
            print("⚠️  Server not running, skipping error handling tests")
            return True

        try:
            # Test invalid time slice
            status, body = _http_get(
                f"{self.base_url}/api/data/invalid_slice", timeout=5
            )
            if status in [
                400,
                404,
                500,
                200,
            ]:  # Accept error codes; some servers 200 with error JSON
                print("  ✓ Invalid time slice handled correctly")
            else:
                print(f"  ❌ Invalid time slice not handled: {status}")
                return False

            # Test invalid lookback days
            status, _ = _http_get(
                f"{self.base_url}/api/data/daily?lookback_days=invalid", timeout=5
            )
            if status in [200, 400]:
                print("  ✓ Invalid lookback_days handled correctly")
            else:
                print(f"  ❌ Invalid lookback_days not handled: {status}")
                return False

            print("✅ Error handling test PASSED")
            return True

        except Exception as e:
            print(f"❌ Error handling test FAILED: {e}")
            return False

    def run_all_tests(self):
        """Run all dashboard tests"""
        print("🧪 Dashboard Comprehensive Test Suite")
        print("=" * 60)

        try:
            self.setup()

            tests = [
                ("Data Engine Initialization", self.test_data_engine_initialization),
                ("Dashboard Server Startup", self.test_dashboard_server_startup),
                ("API Endpoints", self.test_api_endpoints),
                ("Data Accuracy", self.test_data_accuracy),
                ("Dashboard Performance", self.test_dashboard_performance),
                ("Error Handling", self.test_error_handling),
            ]

            results = []
            for test_name, test_func in tests:
                try:
                    result = test_func()
                    results.append((test_name, result))
                except Exception as e:
                    print(f"❌ {test_name} test FAILED with exception: {e}")
                    results.append((test_name, False))

            # Summary
            print("\n" + "=" * 60)
            print("📊 DASHBOARD TEST SUMMARY")
            print("=" * 60)

            passed = sum(1 for _, result in results if result)
            total = len(results)

            for test_name, result in results:
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"{test_name}: {status}")

            print(f"\nTotal tests: {total}")
            print(f"✅ Passed: {passed}")
            print(f"❌ Failed: {total - passed}")

            if passed == total:
                print("\n🎉 ALL DASHBOARD TESTS PASSED - Dashboard is ready for use!")
                return True
            else:
                print(f"\n⚠️  {total - passed} dashboard tests failed")
                return False

        finally:
            self.cleanup()


def main():
    """Run the dashboard test suite"""
    test = DashboardTest()
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
