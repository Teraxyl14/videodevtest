import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), "autonomous_trend_agent"))

from autonomous_trend_agent.captions.animated_captions import AnimatedCaptionEngine

def test_neuro_sync():
    print("Testing Neuro-Sync Engine...")
    
    # Initialize implementation of Hormozi style (Pop + NPO)
    engine = AnimatedCaptionEngine(style="hormozi")
    print(f"Style: {engine.style.name}")
    print(f"NPO Config: {engine.style.npo}ms")
    
    # Dummy data: "Hello World" at 1.0s
    words = [
        {"word": "Hello", "start_time": 1.0, "end_time": 1.5},
        {"word": "World", "start_time": 1.5, "end_time": 2.0}
    ]
    
    # Generate ASS
    output_file = "verify_neuro.ass"
    engine.generate_ass(words, output_file, 1080, 1920)
    
    # Read back and inspect
    with open(output_file, "r") as f:
        content = f.read()
        
    print("\n--- Generated ASS Content ---")
    print(content)
    
    # Verification Logic
    # 1.0s - 50ms = 0.95s -> 0:00:00.95
    if "0:00:00.95" in content:
        print("\n✅ PASS: Negative Perceptual Offset applied (1.0 -> 0.95)")
    else:
        print("\n❌ FAIL: Timestamp not offset correctly")
        
    # Check physics tags
    if "\\fscx80\\fscy80\\t(0,50,\\fscx110\\fscy110)" in content:
        print("✅ PASS: Elastic Pop physics tags found")
    else:
        print("❌ FAIL: Physics tags missing or incorrect")

if __name__ == "__main__":
    test_neuro_sync()
