import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_assessment():
    """Test the risk assessment endpoint"""
    
    # Prepare test data
    test_data = {
        "subject_id": "TEST_001",
        "age": 72,
        "sex": "Male", 
        "mmse": 26.0,
        "feature_values": {
            "Hippocampal_volume": 0.78,
            "Entorhinal_volume": 0.82,
            "Mean_diffusivity_global": 1.15,
            "Fractional_anisotropy_global": 0.68
        }
    }
    
    try:
        print("🚀 Sending assessment request...")
        response = requests.post(f"{BASE_URL}/assess-risk/", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Assessment successful!")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"💥 Connection error: {e}")

def test_health():
    """Test service health status"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print("Service status:", response.json())
    except Exception as e:
        print(f"Service not running: {e}")

if __name__ == "__main__":
    print("🧪 Testing Alzheimer's Disease Risk Assessment API...")
    test_health()
    print("\n" + "="*50 + "\n")
    test_assessment()