# sensors/data_normalizer.py
import redis
import json
import math

class DataNormalizer:
    def __init__(self, host='localhost', port=6379):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.input_stream = 'trend_events'
        self.output_stream = 'normalized_trends'

    def calculate_z_score(self, velocity, platform_weight=1.0):
        # Simplified Z-Score logic
        # In prod: Maintain a moving average of velocities in Redis to compute std_dev
        mean_velocity = 0.5
        std_dev = 0.2
        return (velocity - mean_velocity) / std_dev * platform_weight

    def process_stream(self):
        print("Listening for raw trends...")
        last_id = '$'
        
        while True:
            # Block for new items
            streams = self.r.xread({self.input_stream: last_id}, count=1, block=0)
            
            for stream_name, events in streams:
                for event_id, trend_data in events:
                    last_id = event_id
                    
                    query = trend_data.get('query')
                    velocity = float(trend_data.get('velocity', 0.5))
                    platform = trend_data.get('platform')
                    
                    # Weighting
                    weight = 1.2 if platform == 'tiktok' else 1.0
                    z_score = self.calculate_z_score(velocity, weight)
                    
                    if z_score > 1.5: # "Super Trend" Threshold
                        print(f"SUPER TREND DETECTED: {query} (Score: {z_score:.2f})")
                        self.r.xadd(self.output_stream, {
                            "query": query,
                            "score": z_score,
                            "original_source": platform
                        })

if __name__ == "__main__":
    normalizer = DataNormalizer()
    normalizer.process_stream()
