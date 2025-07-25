"""
Continuous Vehicle Simulation for Docker
Runs vehicle simulation continuously for Docker deployment
"""

import time
import os
import sys
import signal
from datetime import datetime

# Add paths
sys.path.append('/app/storage_utils')
sys.path.append('/app/flink_processor')

from vehicle_simulation import VehicleSimulation

class ContinuousSimulation:
    """Runs vehicle simulation continuously"""
    
    def __init__(self):
        self.running = True
        self.simulation = VehicleSimulation(num_vehicles=5, simulation_duration=60)
        
        # Handle shutdown signals
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Handle shutdown gracefully"""
        print(f"\n🛑 Received shutdown signal {signum}")
        self.running = False
    
    def run(self):
        """Run continuous simulation"""
        print("🚗 Starting Continuous Vehicle Simulation for Docker")
        print("=" * 50)
        
        cycle = 1
        while self.running:
            try:
                print(f"\n📊 Simulation Cycle {cycle}")
                print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run simulation cycle
                raw_data, processed_data = self.simulation.run_simulation(real_time=False)
                
                print(f"✅ Cycle {cycle} completed:")
                print(f"   • Generated: {len(raw_data)} raw records")
                print(f"   • Processed: {len(processed_data)} processed records")
                
                cycle += 1
                
                # Wait before next cycle (5 minutes)
                if self.running:
                    print(f"⏳ Waiting 5 minutes before next cycle...")
                    for i in range(300):  # 5 minutes = 300 seconds
                        if not self.running:
                            break
                        time.sleep(1)
                        if i % 60 == 0:  # Print every minute
                            remaining = (300 - i) // 60
                            print(f"   ⏰ {remaining} minutes remaining...")
                
            except Exception as e:
                print(f"❌ Error in simulation cycle {cycle}: {e}")
                if self.running:
                    print("⏳ Waiting 30 seconds before retry...")
                    time.sleep(30)
        
        print("\n🛑 Continuous simulation stopped")

if __name__ == "__main__":
    sim = ContinuousSimulation()
    sim.run()
