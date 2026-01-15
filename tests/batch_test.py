import requests
import json
import time
import os
import tempfile
from datetime import datetime

BASE_URL = "http://localhost:8000"

# ============================================
# HEALTH CHECK
# ============================================
def test_api_health():
    """Check if API is running"""
    
    print("\n🔍 STEP 0: API Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is responding (Status: {response.status_code})")
            print(f"Response: {result}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"Make sure API is running on {BASE_URL}")
        return False

# ============================================
# TEST 1: Simple JSON Batch (Debug Mode)
# ============================================
def test_json_batch_debug():
    """Test batch prediction with DEBUG output"""
    
    print("\n🧪 TEST 1: JSON Batch (DEBUG MODE)")
    print("=" * 60)
    
    # FIXED: Use 'titles' list only (matches API schema)
    payload = {
        "titles": [
            "Apple releases iPhone 15"
        ]
    }
    
    print(f"📤 Sending payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=payload,
            timeout=10
        )
        
        print(f"\n📥 Response Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        
        else:
            print(f"\n❌ ERROR: Got {response.status_code}")
            print(f"Response text: {response.text}")
            
            try:
                error_json = response.json()
                print(f"\nError details:")
                print(json.dumps(error_json, indent=2))
            except:
                pass
            
            return False
    
    except Exception as e:
        print(f"❌ Request failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# TEST 2: Valid JSON Batch (5 items)
# ============================================
def test_json_batch_valid():
    """Test batch prediction with proper validation"""
    
    print("\n🧪 TEST 2: Valid JSON Batch (5 items)")
    print("=" * 60)
    
    payload = {
        "titles": [
            "Apple releases iPhone 15",
            "Stock market crashes",
            "COVID cases surge",
            "Manchester United wins",
            "New science breakthrough"
        ]
    }
    
    print(f"📤 Sending {len(payload['titles'])} titles...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Status {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        
        print(f"✅ Status: {response.status_code}")
        print(f"📊 Total items: {result.get('count', 'N/A')}")
        
        if 'latency_ms' in result:
            print(f"⏱️  Time: {result['latency_ms']:.0f}ms")
        
        print(f"\n📋 PREDICTIONS:")
        if 'predictions' in result:
            for i, pred in enumerate(result['predictions'][:5]):
                if isinstance(pred, dict):
                    topic = pred.get('topic', 'N/A')
                    confidence = pred.get('confidence', 0)
                    print(f"   {i+1}. {topic:<15} (conf={confidence:.2f})")
        
        print(f"\n✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# TEST 3: Production Mode (No Metadata)
# ============================================
def test_production_mode():
    """Test with just titles (production use case)"""
    
    print("\n🧪 TEST 3: Production Mode (Just Titles)")
    print("=" * 60)
    
    payload = {
        "titles": [
            "Breaking news about technology sector",
            "Market analysis for today",
            "Health guidelines updated"
        ]
    }
    
    print(f"📤 Sending {len(payload['titles'])} titles...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Status {response.status_code}")
            return False
        
        result = response.json()
        
        print(f"✅ Status: {response.status_code}")
        print(f"📊 Predictions received: {result.get('count', 'N/A')}")
        
        print(f"\n📋 PREDICTIONS:")
        if 'predictions' in result:
            for i, pred in enumerate(result['predictions']):
                if isinstance(pred, dict):
                    topic = pred.get('topic', 'N/A')
                    confidence = pred.get('confidence', 0)
                    print(f"   {i+1}. {topic:<15} (conf={confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ============================================
# TEST 4: Large Batch (100 items)
# ============================================
def test_large_batch():
    """Test with 100 items"""
    
    print("\n🧪 TEST 4: Large Batch (100 items)")
    print("=" * 60)
    
    titles = [
        f"Sample news article number {i} about recent developments in the world"
        for i in range(100)
    ]
    
    payload = {"titles": titles}
    
    print(f"📤 Sending {len(titles)} titles...")
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start
        
        if response.status_code != 200:
            print(f"❌ Status {response.status_code}")
            return False
        
        result = response.json()
        
        print(f"✅ Status: {response.status_code}")
        print(f"📊 Items processed: {result.get('count', 'N/A')}/100")
        print(f"⏱️  Total time: {elapsed:.1f}s")
        print(f"⏱️  Avg per item: {elapsed*1000/100:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ============================================
# TEST 5: CSV File Upload
# ============================================
def test_csv_upload():
    """Test CSV file upload"""
    
    print("\n🧪 TEST 5: CSV File Upload")
    print("=" * 60)
    
    # Check if batch_test_examples.csv exists
    csv_file = 'data/batch_test_examples.csv'
    
    if not os.path.exists(csv_file):
        print(f"⚠️  File not found: {csv_file}")
        print(f"   Looking in: {os.path.abspath(csv_file)}")
        
        # Create a sample CSV if it doesn't exist
        print(f"\n📝 Creating sample CSV...")
        
        os.makedirs('data', exist_ok=True)
        
        sample_csv = """title
Apple releases iPhone 15
Stock market crashes today
COVID cases surge in winter
Manchester United wins trophy
Scientists discover new species
New government policy announced
International trade agreement signed
Technology company reports earnings"""
        
        with open(csv_file, 'w') as f:
            f.write(sample_csv)
        
        print(f"✅ Sample CSV created at: {csv_file}")
    
    print(f"📄 Uploading: {csv_file}")
    
    try:
        with open(csv_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/predict/batch/from-csv",
                files=files,
                timeout=60
            )
        
        if response.status_code != 200:
            print(f"❌ Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        
        print(f"✅ Status: {response.status_code}")
        
        # Parse response (could be different formats)
        if 'batch_response' in result:
            batch_resp = result['batch_response']
            print(f"📊 Items processed: {batch_resp.get('successful', 'N/A')}")
        elif 'count' in result:
            print(f"📊 Items processed: {result['count']}")
        
        if 'output_file' in result:
            print(f"💾 Output file: {result['output_file']}")
        
        print(f"\n✅ CSV upload test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# TEST 6: Edge Cases
# ============================================
def test_edge_cases():
    """Test edge cases"""
    
    print("\n🧪 TEST 6: Edge Cases")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Empty title",
            "payload": {"titles": [""]}
        },
        {
            "name": "Very long title",
            "payload": {"titles": ["A" * 500]}
        },
        {
            "name": "Special characters",
            "payload": {"titles": ["Breaking! @#$%^&*() News 🔥"]}
        },
        {
            "name": "Unicode",
            "payload": {"titles": ["新闻 مخبر뉴스 ข่าว"]}
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        name = test_case["name"]
        payload = test_case["payload"]
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict/batch",
                json=payload,
                timeout=10
            )
            
            status = "✅" if response.status_code == 200 else "❌"
            results[name] = response.status_code == 200
            print(f"{status} {name}: {response.status_code}")
        
        except Exception as e:
            results[name] = False
            print(f"❌ {name}: {e}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n📊 Edge cases: {passed}/{total} passed")
    return passed == total

# ============================================
# RUN ALL TESTS
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 BATCH PREDICTION TESTING SUITE")
    print("="*60)
    print(f"API URL: {BASE_URL}")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*60)
    
    try:
        # Health check first
        if not test_api_health():
            print("\n⛔ STOP: API not running or not responding!")
            print(f"Please start the API on {BASE_URL}")
            exit(1)
        
        # Run all tests
        results = {
            "Debug (simple)": test_json_batch_debug(),
            "Valid batch (5 items)": test_json_batch_valid(),
            "Production mode": test_production_mode(),
            "Large batch (100)": test_large_batch(),
            "CSV upload": test_csv_upload(),
            "Edge cases": test_edge_cases()
        }
        
        # Summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, passed_flag in results.items():
            status = "✅" if passed_flag else "❌"
            print(f"{status} {test_name}")
        
        print("\n" + "="*60)
        if passed == total:
            print(f"🎉 ALL {total} TESTS PASSED!")
        else:
            print(f"⚠️  {passed}/{total} tests passed")
        print("="*60)
    
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)