"""
Sheep Detection Module
Uses basic image analysis to detect if image contains a sheep-like animal
"""

import cv2
import numpy as np
from PIL import Image

def detect_sheep_simple(image_path):
    """
    Simple sheep detection using color and texture analysis
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict with:
            - is_sheep: bool (True if likely contains sheep)
            - confidence: float (0-100)
            - reason: str (why it thinks it's sheep or not)
    """
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                'is_sheep': False,
                'confidence': 0,
                'reason': 'Could not load image'
            }
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = img.shape[:2]
        total_pixels = height * width
        
        # Check 1: Detect white/cream colors (sheep wool)
        # Sheep typically have white, cream, or light gray wool
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_pixels = np.sum(white_mask > 0)
        white_percentage = (white_pixels / total_pixels) * 100
        
        # Check 2: Detect brown/tan colors (sheep face/legs)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_pixels = np.sum(brown_mask > 0)
        brown_percentage = (brown_pixels / total_pixels) * 100
        
        # Check 3: Detect texture (wool has distinct texture)
        # Use edge detection to find texture
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        edge_percentage = (edge_pixels / total_pixels) * 100
        
        # Check 4: Detect if image has animal-like shapes
        # Look for contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            contour_percentage = (contour_area / total_pixels) * 100
        else:
            contour_percentage = 0
        
        # Scoring system (IMPROVED - Works for all sheep colors!)
        score = 0
        reasons = []
        
        # Calculate total "sheep color" (white OR brown)
        sheep_color_total = white_percentage + brown_percentage
        
        # Check 1: Has sheep-like colors (white, brown, or both)
        if sheep_color_total > 40:
            score += 40
            reasons.append(f"Contains sheep-like colors (white: {white_percentage:.1f}%, brown: {brown_percentage:.1f}%)")
        elif sheep_color_total > 25:
            score += 25
            reasons.append(f"Contains some sheep-like colors (white: {white_percentage:.1f}%, brown: {brown_percentage:.1f}%)")
        elif sheep_color_total > 15:
            score += 10
            reasons.append(f"Contains minimal sheep-like colors ({sheep_color_total:.1f}%)")
        else:
            score -= 20
            reasons.append(f"Lacks sheep-like colors ({sheep_color_total:.1f}%)")
        
        # Check 2: Texture (wool is textured, not smooth)
        if 8 < edge_percentage < 35:
            score += 30
            reasons.append(f"Has wool-like texture ({edge_percentage:.1f}%)")
        elif edge_percentage >= 35:
            score += 15
            reasons.append(f"Has texture but very detailed ({edge_percentage:.1f}%)")
        elif edge_percentage < 5:
            score -= 30
            reasons.append(f"Too smooth - sheep have textured wool ({edge_percentage:.1f}%)")
        else:
            score += 10
            reasons.append(f"Has some texture ({edge_percentage:.1f}%)")
        
        # Check 3: Has defined shape (animal contour)
        if contour_percentage > 15:
            score += 20
            reasons.append(f"Has clear animal shape ({contour_percentage:.1f}%)")
        elif contour_percentage > 8:
            score += 10
            reasons.append(f"Has some defined shape ({contour_percentage:.1f}%)")
        
        # Check 4: Color distribution (sheep have varied colors, not uniform)
        color_variety = max(white_percentage, brown_percentage) - min(white_percentage, brown_percentage)
        if 10 < color_variety < 60:
            score += 10
            reasons.append(f"Good color variety (typical of sheep)")
        
        # Penalty checks
        # Too uniform (blank image or single color background)
        if white_percentage > 85 or brown_percentage > 85:
            score -= 40
            reasons.append("Too uniform - likely not a sheep")
        
        # Human skin detection (peachy/pink tones with low texture)
        # Detect skin-like colors (hue in pink/peach range)
        lower_skin = np.array([0, 20, 100])
        upper_skin = np.array([25, 150, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = np.sum(skin_mask > 0)
        skin_percentage = (skin_pixels / total_pixels) * 100
        
        if skin_percentage > 15 and edge_percentage < 15:
            score -= 35
            reasons.append(f"Detected human skin tones ({skin_percentage:.1f}%) with smooth texture")
        
        # Very low texture = smooth object (not sheep)
        if edge_percentage < 5:
            score -= 25
            reasons.append("Extremely smooth surface - sheep have textured wool")
        
        # Final decision
        is_sheep = score >= 45  # Balanced threshold
        confidence = min(max(score, 0), 100)
        
        if is_sheep:
            reason = "Likely contains a sheep: " + "; ".join(reasons)
        else:
            reason = "Does not appear to contain a sheep: " + "; ".join(reasons)
        
        return {
            'is_sheep': is_sheep,
            'confidence': confidence,
            'reason': reason,
            'details': {
                'white_percentage': round(white_percentage, 2),
                'brown_percentage': round(brown_percentage, 2),
                'sheep_color_total': round(sheep_color_total, 2),
                'skin_percentage': round(skin_percentage, 2),
                'edge_percentage': round(edge_percentage, 2),
                'contour_percentage': round(contour_percentage, 2),
                'score': score
            }
        }
        
    except Exception as e:
        return {
            'is_sheep': False,
            'confidence': 0,
            'reason': f'Error analyzing image: {str(e)}'
        }


# Test the detector
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = detect_sheep_simple(image_path)
        
        print("\n" + "="*60)
        print("🐑 SHEEP DETECTION TEST")
        print("="*60)
        print(f"\nImage: {image_path}")
        print(f"\nIs Sheep: {'✅ YES' if result['is_sheep'] else '❌ NO'}")
        print(f"Confidence: {result['confidence']}/100")
        print(f"\nReason: {result['reason']}")
        print(f"\nDetails:")
        for key, value in result['details'].items():
            print(f"  {key}: {value}")
        print("="*60)
    else:
        print("Usage: python utils/sheep_detector.py <image_path>")