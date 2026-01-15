# Run this NOW to validate your model!

import requests
import json

BASE_URL = "http://localhost:8000"

test_cases = [
    # EASY: Clear single-topic
    ("Manchester United defeats Liverpool 3-2 in penalty shootout", "SPORTS"),
    ("Apple launches new iPhone 15 with AI features", "TECHNOLOGY"),
    ("Federal Reserve raises interest rates by 0.25%", "BUSINESS"),
    
    # MEDIUM: Potential overlap
    ("New COVID-19 vaccine approved by FDA", "HEALTH"),
    ("Scientists discover new species in Amazon rainforest", "SCIENCE"),
    ("Tom Cruise stars in new action movie released this week", "ENTERTAINMENT"),
    
    # HARD: Ambiguous
    ("US Senate votes on new healthcare bill", "NATION"),
    ("China-US trade tensions escalate over tech exports", "WORLD"),
    ("Stock market crashes 5% amid inflation concerns", "BUSINESS"),
]

print("🧪 VALIDATING YOUR MODEL (9 Test Cases)")
print("=" * 80)

results = []
correct = 0

for title, expected in test_cases:
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"title": title},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        data = response.json()
        predicted = data['topic']
        confidence = data['confidence']
        latency = data['latency_ms']
        
        is_correct = predicted == expected
        correct += is_correct
        status = "✅" if is_correct else "❌"
        
        results.append({
            'title': title[:50],
            'expected': expected,
            'predicted': predicted,
            'confidence': f"{confidence:.1%}",
            'latency': f"{latency:.0f}ms",
            'status': status
        })
        
        print(f"{status} {expected:<12} → {predicted:<12} ({confidence:.1%}) [{latency:.0f}ms]")
        
    except Exception as e:
        print(f"⚠️  ERROR: {title[:40]}... - {str(e)}")

accuracy = correct / len(test_cases)
print("\n" + "=" * 80)
print(f"🎯 ACCURACY: {correct}/{len(test_cases)} = {accuracy:.1%}")

if accuracy >= 0.89:
    print("🟢 EXCELLENT! Model is production-ready!")
elif accuracy >= 0.77:
    print("🟡 GOOD! Small improvements needed")
else:
    print("🔴 ISSUES REMAIN - Debug needed")