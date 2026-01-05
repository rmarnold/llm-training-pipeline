"""Background GPU keepalive to prevent Colab idle timeout.

Run in a separate cell with:
    import subprocess
    subprocess.Popen(['python', 'scripts/gpu_keepalive.py'])

Or inline:
    !nohup python scripts/gpu_keepalive.py &
"""

import torch
import time
import os
import signal
import sys

# Configuration
IDLE_CHECK_INTERVAL = 60  # Check every 60 seconds
KEEPALIVE_THRESHOLD = 300  # Spike GPU if idle for 5 minutes (300 seconds)
SPIKE_DURATION = 0.5  # Brief GPU computation

# Track last GPU activity
last_gpu_active = time.time()


def gpu_spike():
    """Perform a brief GPU computation to prevent idle timeout."""
    if not torch.cuda.is_available():
        return False

    try:
        # Small matrix multiplication - enough to register as GPU activity
        device = torch.device('cuda')
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        del a, b, c
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"GPU spike failed: {e}")
        return False


def check_gpu_idle():
    """Check if GPU has been idle (no processes using it)."""
    try:
        # Check nvidia-smi for GPU utilization
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        utilization = int(result.stdout.strip())
        return utilization < 5  # Consider idle if < 5% utilization
    except:
        return True  # Assume idle if check fails


def signal_handler(sig, frame):
    """Handle graceful shutdown."""
    print("\nGPU keepalive stopped.")
    sys.exit(0)


def main():
    print(f"GPU Keepalive started (PID: {os.getpid()})")
    print(f"  - Check interval: {IDLE_CHECK_INTERVAL}s")
    print(f"  - Keepalive threshold: {KEEPALIVE_THRESHOLD}s")
    print(f"  - To stop: kill {os.getpid()}")

    if not torch.cuda.is_available():
        print("WARNING: No GPU available, keepalive will not be effective")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    last_spike = time.time()
    idle_start = None

    while True:
        try:
            time.sleep(IDLE_CHECK_INTERVAL)

            is_idle = check_gpu_idle()

            if is_idle:
                if idle_start is None:
                    idle_start = time.time()

                idle_duration = time.time() - idle_start

                if idle_duration >= KEEPALIVE_THRESHOLD:
                    print(f"[{time.strftime('%H:%M:%S')}] GPU idle for {idle_duration:.0f}s, sending keepalive spike...")
                    if gpu_spike():
                        print(f"[{time.strftime('%H:%M:%S')}] Keepalive spike sent successfully")
                        last_spike = time.time()
                        idle_start = time.time()  # Reset idle timer
            else:
                idle_start = None  # Reset if GPU becomes active

        except Exception as e:
            print(f"Keepalive error: {e}")
            time.sleep(IDLE_CHECK_INTERVAL)


if __name__ == "__main__":
    main()
