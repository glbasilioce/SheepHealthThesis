"""
Overall Health Index (OHI) Calculator
Calculates health score (0-100) based on vital signs
"""

def calculate_parameter_score(value, param_type):
    """
    Calculate score (0-100) for a single parameter
    
    Args:
        value: The measured value
        param_type: Type of parameter (heart_rate, body_temp, spo2, activity_level)
    
    Returns:
        Score from 0-100 (100 = perfect)
    """
    
    # Define normal ranges and thresholds
    thresholds = {
        'heart_rate': {
            'optimal': (70, 90),      # Best range
            'normal': (60, 100),      # Acceptable range
            'mild_concern': (100, 110),
            'critical': 110           # Above this is critical
        },
        'body_temp': {
            'optimal': (38.5, 39.5),
            'normal': (38.0, 40.0),
            'mild_concern': (40.0, 40.5),
            'critical': 40.5
        },
        'spo2': {
            'optimal': 96,            # >= 96 is optimal
            'normal': 92,             # >= 92 is acceptable
            'mild_concern': 90,       # 90-92 is concerning
            'critical': 90            # < 90 is critical
        },
        'activity_level': {
            'optimal': 1.0,           # Normal activity
            'low': 0.8,               # Slightly low
            'very_low': 0.6,          # Concerning
            'critical': 0.5           # Very concerning
        }
    }
    
    thresh = thresholds[param_type]
    
    # Calculate score based on parameter type
    if param_type == 'heart_rate':
        if thresh['optimal'][0] <= value <= thresh['optimal'][1]:
            # Optimal range: 90-100 score
            return 95
        elif thresh['normal'][0] <= value <= thresh['normal'][1]:
            # Normal range: 70-89 score
            return 80
        elif value <= thresh['mild_concern'][1]:
            # Mild concern: 50-69 score
            return 60
        else:
            # Critical: 0-49 score
            excess = value - thresh['critical']
            return max(0, 40 - (excess * 3))
    
    elif param_type == 'body_temp':
        if thresh['optimal'][0] <= value <= thresh['optimal'][1]:
            # Optimal: 90-100
            return 95
        elif thresh['normal'][0] <= value <= thresh['normal'][1]:
            # Normal: 70-89
            return 80
        elif value <= thresh['mild_concern'][1]:
            # Mild concern: 50-69
            return 60
        else:
            # Critical: 0-49
            excess = value - thresh['critical']
            return max(0, 40 - (excess * 10))
    
    elif param_type == 'spo2':
        if value >= thresh['optimal']:
            # Optimal: 90-100
            return 95
        elif value >= thresh['normal']:
            # Normal: 70-89
            return 80
        elif value >= thresh['mild_concern']:
            # Mild concern: 50-69
            return 60
        else:
            # Critical: 0-49
            deficit = thresh['critical'] - value
            return max(0, 40 - (deficit * 5))
    
    elif param_type == 'activity_level':
        if value >= thresh['optimal']:
            # Normal or high activity: 90-100
            return 95
        elif value >= thresh['low']:
            # Slightly low: 70-89
            return 80
        elif value >= thresh['very_low']:
            # Concerning: 50-69
            return 60
        else:
            # Critical: 0-49
            return max(0, int(value / thresh['critical'] * 40))
    
    return 50  # Default middle score


def calculate_ohi(vitals):
    """
    Calculate Overall Health Index
    
    Args:
        vitals: dict with keys:
            - heart_rate (bpm)
            - body_temp (°C)
            - spo2 (%)
            - activity_level (relative)
            - ambient_temp (°C) - optional
            - humidity (%) - optional
    
    Returns:
        dict with:
            - ohi_score: Overall score (0-100)
            - classification: Text classification
            - status: Status level (normal/mild/unhealthy/critical)
            - parameter_scores: Individual scores
            - alerts: List of concerning parameters
    """
    
    # Calculate individual parameter scores
    hr_score = calculate_parameter_score(vitals['heart_rate'], 'heart_rate')
    temp_score = calculate_parameter_score(vitals['body_temp'], 'body_temp')
    spo2_score = calculate_parameter_score(vitals['spo2'], 'spo2')
    activity_score = calculate_parameter_score(vitals['activity_level'], 'activity_level')
    
    # Environmental score (simplified)
    env_score = 85  # Default good
    if 'ambient_temp' in vitals:
        if vitals['ambient_temp'] > 30 or vitals['ambient_temp'] < 15:
            env_score = 70  # Stressful environment
    if 'humidity' in vitals:
        if vitals['humidity'] > 80:
            env_score -= 10
    
    # Weighted OHI calculation
    # Temperature and heart rate are most important
    ohi = (
        temp_score * 0.30 +      # 30% weight
        hr_score * 0.25 +        # 25% weight
        spo2_score * 0.20 +      # 20% weight
        activity_score * 0.15 +  # 15% weight
        env_score * 0.10         # 10% weight
    )
    
    # Classification based on OHI score
    if ohi >= 85:
        classification = "Healthy"
        status = "normal"
        status_color = "success"
    elif ohi >= 70:
        classification = "Mild Risk"
        status = "mild"
        status_color = "warning"
    elif ohi >= 50:
        classification = "Unhealthy"
        status = "unhealthy"
        status_color = "warning"
    else:
        classification = "Critical"
        status = "critical"
        status_color = "danger"
    
    # Generate alerts for concerning parameters
    alerts = []
    if temp_score < 70:
        alerts.append(f"⚠️ Body temperature abnormal: {vitals['body_temp']}°C")
    if hr_score < 70:
        alerts.append(f"⚠️ Heart rate abnormal: {vitals['heart_rate']} bpm")
    if spo2_score < 70:
        alerts.append(f"⚠️ Blood oxygen low: {vitals['spo2']}%")
    if activity_score < 70:
        alerts.append(f"⚠️ Activity level low: {vitals['activity_level']}")
    
    return {
        'ohi_score': round(ohi, 2),
        'classification': classification,
        'status': status,
        'status_color': status_color,
        'parameter_scores': {
            'heart_rate': round(hr_score, 2),
            'body_temp': round(temp_score, 2),
            'spo2': round(spo2_score, 2),
            'activity': round(activity_score, 2),
            'environment': round(env_score, 2)
        },
        'alerts': alerts
    }


# Test the calculator
if __name__ == "__main__":
    print("🧪 TESTING OHI CALCULATOR")
    print("=" * 60)
    
    # Test 1: Healthy sheep
    print("\n✅ TEST 1: HEALTHY SHEEP")
    healthy_vitals = {
        'heart_rate': 85,
        'body_temp': 39.2,
        'spo2': 97,
        'activity_level': 1.0,
        'ambient_temp': 25,
        'humidity': 60
    }
    
    result = calculate_ohi(healthy_vitals)
    print(f"   OHI Score: {result['ohi_score']}/100")
    print(f"   Classification: {result['classification']}")
    print(f"   Status: {result['status']}")
    print(f"   Parameter Scores:")
    for param, score in result['parameter_scores'].items():
        print(f"      {param}: {score}/100")
    if result['alerts']:
        print(f"   Alerts: {result['alerts']}")
    else:
        print(f"   Alerts: None")
    
    # Test 2: Sick sheep (fever)
    print("\n🔥 TEST 2: SICK SHEEP (Fever)")
    sick_vitals = {
        'heart_rate': 115,
        'body_temp': 40.8,
        'spo2': 88,
        'activity_level': 0.5,
        'ambient_temp': 28,
        'humidity': 75
    }
    
    result = calculate_ohi(sick_vitals)
    print(f"   OHI Score: {result['ohi_score']}/100")
    print(f"   Classification: {result['classification']}")
    print(f"   Status: {result['status']}")
    print(f"   Parameter Scores:")
    for param, score in result['parameter_scores'].items():
        print(f"      {param}: {score}/100")
    if result['alerts']:
        print(f"   Alerts:")
        for alert in result['alerts']:
            print(f"      {alert}")
    
    # Test 3: Mild concern
    print("\n⚠️  TEST 3: MILD CONCERN")
    mild_vitals = {
        'heart_rate': 102,
        'body_temp': 40.2,
        'spo2': 93,
        'activity_level': 0.85,
        'ambient_temp': 26,
        'humidity': 65
    }
    
    result = calculate_ohi(mild_vitals)
    print(f"   OHI Score: {result['ohi_score']}/100")
    print(f"   Classification: {result['classification']}")
    print(f"   Status: {result['status']}")
    if result['alerts']:
        print(f"   Alerts:")
        for alert in result['alerts']:
            print(f"      {alert}")
    else:
        print(f"   Alerts: None")
    
    print("\n✅ OHI Calculator working!")
    print("=" * 60)