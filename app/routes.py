"""
Flask Routes
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime

# App modules
from app.predict import predict_disease
from app.models import db, Prediction, HealthReading
from app.disease_info import get_disease_info

# Utils modules
from utils.ohi_calculator import calculate_ohi
from utils.sheep_detector import detect_sheep_simple
from utils.sensor_simulator import SensorSimulator

# ✨ Create simulator once (global, reused for all requests)
simulator = SensorSimulator()

bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('main.index'))
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('main.index'))
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)', 'danger')
        return redirect(url_for('main.index'))
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join('app/static/uploads', filename)
        file.save(filepath)
        
        # Check if image contains a sheep
        sheep_check = detect_sheep_simple(filepath)

        # Block if not a sheep (threshold: 45%)
        if sheep_check['confidence'] < 45:
            # Delete uploaded file (cleanup)
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Show error message
            flash('❌ Image Validation Failed', 'danger')
            flash(f'The uploaded image does not appear to contain a sheep (Confidence: {sheep_check["confidence"]:.0f}%).', 'warning')
            flash('Please upload a clear image of a sheep.', 'info')
            flash(f'Reason: {sheep_check["reason"]}', 'secondary')
            
            return redirect(url_for('main.index'))

        result = predict_disease(filepath)
        result['sheep_detection'] = sheep_check
        
        # Add warning if sheep detection is uncertain
        if sheep_check['confidence'] < 60:
            result['sheep_info'] = f'Sheep detected with moderate confidence ({sheep_check["confidence"]:.0f}%). Results may vary.'
            result['sheep_info_level'] = 'moderate'
        elif sheep_check['confidence'] < 80:
            result['sheep_info'] = f'Sheep detected with good confidence ({sheep_check["confidence"]:.0f}%).'
            result['sheep_info_level'] = 'good'
        else:
            result['sheep_info'] = f'Sheep detected with high confidence ({sheep_check["confidence"]:.0f}%)!'
            result['sheep_info_level'] = 'excellent'

        result['warning'] = None
        result['warning_level'] = 'low'

        # Get disease details
        result['disease_details'] = get_disease_info(result['predicted_class'])
        
        # Save to database
        prediction_record = Prediction(
            filename=filename,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            all_predictions=result['all_predictions']
        )
        db.session.add(prediction_record)
        db.session.commit()
        
        # Render result page
        return render_template('result.html', 
                             result=result,
                             image_path=f'uploads/{filename}')
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'danger')
        return redirect(url_for('main.index'))
    
@bp.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html')

@bp.route('/history')
def history():
    """View prediction history"""
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(50).all()
    return render_template('history.html', predictions=predictions)

@bp.route('/health-monitor')
def health_monitor():
    """Health monitoring dashboard"""
    return render_template('health_monitor.html')

@bp.route('/api/sensor-data')
def get_sensor_data():
    """API endpoint to get current sensor readings"""
    
    # ✨ Change health state for testing
    # Uncomment ONE of these lines to test different states:
    simulator.set_health_state('normal')
    # simulator.set_health_state('fever')
    # simulator.set_health_state('respiratory')
    # simulator.set_health_state('critical')
    
    # Get reading
    reading = simulator.get_reading()
    
    # Calculate OHI
    ohi_result = calculate_ohi(reading)
    
    # Save to database
    health_reading = HealthReading(
        sheep_id=reading['sheep_id'],
        heart_rate=reading['heart_rate'],
        body_temp=reading['body_temp'],
        spo2=reading['spo2'],
        activity_level=reading['activity_level'],
        ambient_temp=reading['ambient_temp'],
        humidity=reading['humidity'],
        ohi_score=ohi_result['ohi_score'],
        health_status=ohi_result['status']
    )
    db.session.add(health_reading)
    db.session.commit()

    # Combine data
    response = {
        'vitals': reading,
        'ohi': ohi_result,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return json.dumps(response)

@bp.route('/health-history')
def health_history():
    """Health history page with charts"""
    return render_template('health_history.html')

@bp.route('/api/health-history')
def get_health_history():
    """API endpoint for health history data"""
    
    # Get last 50 readings
    readings = HealthReading.query.order_by(HealthReading.timestamp.desc()).limit(50).all()
    readings_list = [r.to_dict() for r in reversed(readings)]
    
    # Calculate summary stats
    if readings:
        avg_ohi = sum(r.ohi_score for r in readings) / len(readings)
        avg_temp = sum(r.body_temp for r in readings) / len(readings)
        avg_hr = sum(r.heart_rate for r in readings) / len(readings)
    else:
        avg_ohi = 0
        avg_temp = 0
        avg_hr = 0
    
    # Get recent 20 for table
    recent = HealthReading.query.order_by(HealthReading.timestamp.desc()).limit(20).all()
    recent_list = [r.to_dict() for r in recent]
    
    response = {
        'readings': readings_list,
        'recent': recent_list,
        'summary': {
            'avg_ohi': avg_ohi,
            'avg_temp': avg_temp,
            'avg_hr': avg_hr,
            'total_readings': len(readings)
        }
    }
    
    return json.dumps(response)

@bp.route('/export/health-csv')
def export_health_csv():
    """Export health data as CSV"""
    import csv
    from io import StringIO
    from flask import make_response
    
    # Get all readings
    readings = HealthReading.query.order_by(HealthReading.timestamp.desc()).limit(1000).all()
    
    # Create CSV
    si = StringIO()
    writer = csv.writer(si)
    
    # Header
    writer.writerow([
        'Timestamp', 'Sheep ID', 'OHI Score', 'Health Status',
        'Heart Rate (bpm)', 'Body Temp (°C)', 'SpO2 (%)', 
        'Activity Level', 'Ambient Temp (°C)', 'Humidity (%)'
    ])
    
    # Data rows
    for r in readings:
        writer.writerow([
            r.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            r.sheep_id,
            r.ohi_score,
            r.health_status,
            r.heart_rate,
            r.body_temp,
            r.spo2,
            r.activity_level,
            r.ambient_temp,
            r.humidity
        ])
    
    # Create response
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=health_data.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output

@bp.route('/export/health-pdf')
def export_health_pdf():
    """Export health report as PDF"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from io import BytesIO
    from flask import make_response
    
    # Get recent readings
    readings = HealthReading.query.order_by(HealthReading.timestamp.desc()).limit(50).all()
    
    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1
    )
    
    elements.append(Paragraph("🐑 Sheep Health Monitoring Report", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary info
    if readings:
        avg_ohi = sum(r.ohi_score for r in readings) / len(readings)
        avg_temp = sum(r.body_temp for r in readings) / len(readings)
        avg_hr = sum(r.heart_rate for r in readings) / len(readings)
        
        summary_text = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Sheep ID:</b> DEMO_SHEEP<br/>
        <b>Total Readings:</b> {len(readings)}<br/>
        <b>Average OHI Score:</b> {avg_ohi:.1f}/100<br/>
        <b>Average Temperature:</b> {avg_temp:.1f}°C<br/>
        <b>Average Heart Rate:</b> {avg_hr:.0f} bpm<br/>
        """
        elements.append(Paragraph(summary_text, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Recent readings table
        elements.append(Paragraph("<b>Recent Health Readings</b>", styles['Heading2']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Table data
        table_data = [[
            'Time', 'OHI', 'Status', 'HR', 'Temp', 'SpO2'
        ]]
        
        for r in readings[:20]:
            table_data.append([
                r.timestamp.strftime('%m/%d %H:%M'),
                f"{r.ohi_score:.0f}",
                r.health_status,
                f"{r.heart_rate:.0f}",
                f"{r.body_temp:.1f}",
                f"{r.spo2:.0f}"
            ])
        
        # Create table
        t = Table(table_data, colWidths=[1.2*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.7*inch, 0.6*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        elements.append(t)
        
        # Health status summary
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("<b>Health Status Summary</b>", styles['Heading2']))
        
        status_counts = {}
        for r in readings:
            status_counts[r.health_status] = status_counts.get(r.health_status, 0) + 1
        
        status_text = "<br/>".join([
            f"<b>{status.capitalize()}:</b> {count} readings ({count/len(readings)*100:.1f}%)"
            for status, count in status_counts.items()
        ])
        
        elements.append(Paragraph(status_text, styles['Normal']))
    
    else:
        elements.append(Paragraph("No health data available.", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    
    # Create response
    pdf_data = buffer.getvalue()
    buffer.close()
    
    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=health_report.pdf'
    
    return response

# ✨ BONUS: Dynamic health state control (optional!)
@bp.route('/api/set-health-state/<state>')
def set_health_state(state):
    """Change simulator health state without restarting"""
    if state in ['normal', 'fever', 'respiratory', 'critical']:
        simulator.set_health_state(state)
        return json.dumps({
            'status': 'success', 
            'state': state,
            'message': f'Health state changed to {state}'
        })
    return json.dumps({
        'status': 'error', 
        'message': 'Invalid state. Use: normal, fever, respiratory, or critical'
    })