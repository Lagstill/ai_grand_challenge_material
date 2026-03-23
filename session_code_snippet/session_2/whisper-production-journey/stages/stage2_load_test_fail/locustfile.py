"""
Stage 2: The Load Test That Reveals The Truth
==============================================

This Locust file simulates real-world traffic patterns and shows
why the naive implementation fails.

Run with:
    locust -f locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 and start the test.

Recommended test parameters for demo:
- Users: 10 (start small)
- Spawn rate: 2/sec
- Watch p99 latency explode and requests start failing
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import os
import time

# Path to test audio file - supports both local and Docker runs
# In Docker: mounted at /mnt/locust/audio_samples/
# Local: relative to this file
if os.path.exists("/mnt/locust/audio_samples/sample_10s.wav"):
    AUDIO_FILE = "/mnt/locust/audio_samples/sample_10s.wav"
else:
    AUDIO_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "audio_samples", "sample_10s.wav")


class TranscriptionUser(HttpUser):
    """
    Simulates a user making transcription requests.
    
    In real-world scenarios:
    - Users don't wait politely for responses
    - Multiple users hit the API simultaneously
    - Some files are larger than others
    """
    
    # Wait 1-3 seconds between requests (realistic user behavior)
    wait_time = between(1, 3)
    
    @task(10)
    def transcribe_short_audio(self):
        """
        Most common request: short audio file
        
        This is the "happy path" that works in demos but fails at scale.
        """
        with open(AUDIO_FILE, "rb") as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            
            start = time.time()
            with self.client.post(
                "/transcribe",
                files=files,
                catch_response=True,
                timeout=60  # Generous timeout
            ) as response:
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    # Check if response time is acceptable
                    if elapsed > 30:
                        response.failure(f"Too slow: {elapsed:.1f}s")
                    else:
                        response.success()
                elif response.status_code == 503:
                    response.failure("Server overloaded (503)")
                else:
                    response.failure(f"Error: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Occasional health checks - these should always succeed quickly.
        
        In production, your load balancer does this constantly.
        Watch how even health checks slow down under load.
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.elapsed.total_seconds() > 1:
                response.failure("Health check too slow - server struggling")


class AggressiveUser(HttpUser):
    """
    The "impatient" user pattern.
    
    Some users (or automated systems) don't wait between requests.
    This is what happens when someone integrates your API into a batch job.
    """
    
    wait_time = between(0.1, 0.5)  # Rapid fire
    weight = 2  # Less common but still significant
    
    @task
    def rapid_transcribe(self):
        """
        Back-to-back requests with minimal delay.
        
        This pattern breaks naive implementations fastest.
        """
        with open(AUDIO_FILE, "rb") as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            
            with self.client.post(
                "/transcribe",
                files=files,
                catch_response=True,
                timeout=30
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Failed: {response.status_code}")


# Event hooks for custom reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log when test begins"""
    print("\n" + "="*60)
    print("🚀 LOAD TEST STARTING")
    print("="*60)
    print("Watch the following metrics:")
    print("  - p99 latency (should stay under 5-10s)")
    print("  - Failure rate (should stay at 0%)")
    print("  - RPS (requests per second)")
    print("="*60 + "\n")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track individual request metrics for analysis"""
    if response_time > 30000:  # > 30 seconds
        print(f"⚠️ SLOW REQUEST: {name} took {response_time/1000:.1f}s")


# Custom failure thresholds
LATENCY_THRESHOLD_MS = 30000  # 30 second p99 is our target
FAILURE_RATE_THRESHOLD = 0.05  # 5% failure rate max
