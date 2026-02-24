"""
Flask Application Entry Point
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🐑 SHEEP DISEASE DETECTION WEB APP")
    print("="*60)
    print("\n🌐 Starting server...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(
    debug=True, 
    host='0.0.0.0', 
    port=5000,
    use_reloader=False  # ✨ Prevents double process!
)