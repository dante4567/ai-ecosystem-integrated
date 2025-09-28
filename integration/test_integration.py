#!/usr/bin/env python3

"""
Integration Test Script

Tests all components of the AI ecosystem integration
to ensure everything is working correctly.
"""

import asyncio
import aiohttp
import json
import time
import sys
from typing import Dict, Any

# Service URLs
SERVICES = {
    "ChromaDB": "http://localhost:8000/api/v1/heartbeat",
    "RAG Service": "http://localhost:8001/health",
    "Personal Service": "http://localhost:8002/health",
    "Gateway": "http://localhost:8003/health"
}

TESTS = [
    {
        "name": "Gateway Unified Search",
        "method": "POST",
        "url": "http://localhost:8003/api/search",
        "data": {"query": "machine learning", "max_results": 3, "include_personal": True}
    },
    {
        "name": "Gateway Smart Assistant",
        "method": "POST",
        "url": "http://localhost:8003/api/assistant",
        "data": {"message": "What tasks do I have?", "include_actions": True}
    },
    {
        "name": "Gateway Service Status",
        "method": "GET",
        "url": "http://localhost:8003/api/services"
    },
    {
        "name": "RAG Document Search",
        "method": "POST",
        "url": "http://localhost:8001/search",
        "data": {"query": "AI", "max_results": 3}
    },
    {
        "name": "Personal Tasks List",
        "method": "GET",
        "url": "http://localhost:8002/tasks"
    }
]

class IntegrationTester:
    def __init__(self):
        self.session = None
        self.results = []

    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session

    async def test_service_health(self, name: str, url: str) -> Dict[str, Any]:
        """Test individual service health"""
        start_time = time.time()
        try:
            session = await self.get_session()
            async with session.get(url) as response:
                end_time = time.time()
                response_time = end_time - start_time

                if response.status == 200:
                    data = await response.json() if response.content_type == "application/json" else {}
                    return {
                        "service": name,
                        "status": "PASS",
                        "response_time": f"{response_time:.3f}s",
                        "details": data
                    }
                else:
                    return {
                        "service": name,
                        "status": "FAIL",
                        "error": f"HTTP {response.status}",
                        "response_time": f"{response_time:.3f}s"
                    }
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return {
                "service": name,
                "status": "FAIL",
                "error": str(e),
                "response_time": f"{response_time:.3f}s"
            }

    async def run_test(self, test: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single integration test"""
        start_time = time.time()
        try:
            session = await self.get_session()
            kwargs = {}
            if test.get("data"):
                kwargs["json"] = test["data"]

            async with session.request(test["method"], test["url"], **kwargs) as response:
                end_time = time.time()
                response_time = end_time - start_time

                if response.status == 200:
                    data = await response.json() if response.content_type == "application/json" else {}
                    return {
                        "test": test["name"],
                        "status": "PASS",
                        "response_time": f"{response_time:.3f}s",
                        "data_size": len(str(data))
                    }
                else:
                    error_text = await response.text()
                    return {
                        "test": test["name"],
                        "status": "FAIL",
                        "error": f"HTTP {response.status}: {error_text[:100]}",
                        "response_time": f"{response_time:.3f}s"
                    }
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return {
                "test": test["name"],
                "status": "FAIL",
                "error": str(e),
                "response_time": f"{response_time:.3f}s"
            }

    async def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 AI Ecosystem Integration Tests")
        print("=" * 50)

        # Test service health first
        print("\n🏥 Service Health Checks:")
        health_results = []
        for service_name, service_url in SERVICES.items():
            result = await self.test_service_health(service_name, service_url)
            health_results.append(result)

            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"  {status_emoji} {service_name}: {result['status']} ({result['response_time']})")

        # Count healthy services
        healthy_services = sum(1 for r in health_results if r["status"] == "PASS")
        total_services = len(health_results)

        print(f"\n📊 Health Summary: {healthy_services}/{total_services} services healthy")

        if healthy_services < total_services:
            print("⚠️  Some services are down. Integration tests may fail.")
            print("   Make sure all services are running: docker-compose up")

        # Run integration tests
        print("\n🔗 Integration Tests:")
        test_results = []
        for test in TESTS:
            result = await self.run_test(test)
            test_results.append(result)

            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"  {status_emoji} {test['name']}: {result['status']} ({result['response_time']})")

            if result["status"] == "FAIL":
                print(f"     Error: {result.get('error', 'Unknown error')}")

        # Summary
        passed_tests = sum(1 for r in test_results if r["status"] == "PASS")
        total_tests = len(test_results)

        print(f"\n📊 Test Summary: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests and healthy_services == total_services:
            print("🎉 All tests passed! Integration is working correctly.")
            return True
        else:
            print("❌ Some tests failed. Check service logs for details:")
            print("   docker-compose logs [service-name]")
            return False

    async def close(self):
        if self.session:
            await self.session.close()

async def main():
    """Main test function"""
    tester = IntegrationTester()
    try:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
    finally:
        await tester.close()

if __name__ == "__main__":
    print("Starting AI Ecosystem Integration Tests...")
    print("Make sure all services are running: docker-compose up")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)