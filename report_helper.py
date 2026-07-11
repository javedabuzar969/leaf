import os
import io
import datetime
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def generate_pdf_report(
    diagnosis_name, 
    confidence, 
    severity, 
    severity_color, 
    original_np, 
    gradcam_np, 
    soil_data, 
    weather_data, 
    approved_recommendation
):
    """
    Generate a professional PDF report containing the diagnosis, images, soil/weather data, 
    and the final approved recommendation. Returns a bytes object for download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    story = []
    
    # Styles Setup
    styles = getSampleStyleSheet()
    
    # Custom colors
    primary_color = colors.HexColor("#1e4620")  # Dark forest green
    secondary_color = colors.HexColor("#eceff1") # Light grey for tables
    text_color = colors.HexColor("#263238")      # Dark grey/black
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.white,
        alignment=TA_CENTER,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    
    heading1_style = ParagraphStyle(
        'Heading1Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        spaceAfter=6
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyStyle',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    rec_style = ParagraphStyle(
        'RecStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=text_color,
        leading=13.5,
        spaceAfter=4
    )
    
    rec_heading_style = ParagraphStyle(
        'RecHeadingStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    # 1. Header Banner Table
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    header_content = [
        [Paragraph("AGRICULTURE ADVISOR AI CO-PILOT", title_style)],
        [Paragraph(f"Agronomic Diagnostic Report &bull; Generated: {date_str}", subtitle_style)]
    ]
    header_table = Table(header_content, colWidths=[532])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), primary_color),
        ('PADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,1), (-1,1), 12),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 15))
    
    # 2. Diagnosis & Metadata
    sev_color_hex = colors.HexColor(severity_color)
    diag_summary = [
        [Paragraph("Target Crop / Disease:", bold_body_style), Paragraph(diagnosis_name, body_style)],
        [Paragraph("CNN Prediction Confidence:", bold_body_style), Paragraph(f"{confidence*100:.1f}%", body_style)],
        [Paragraph("Severity Class Assessment:", bold_body_style), Paragraph(f"<font color='{severity_color}'><b>{severity}</b></font>", body_style)],
        [Paragraph("Region / Farm Location:", bold_body_style), Paragraph(weather_data.get('city', 'N/A'), body_style)]
    ]
    diag_table = Table(diag_summary, colWidths=[180, 352])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 15))
    
    # 3. Save images to disk temporarily to put in PDF
    temp_orig_path = "temp_orig_pdf.png"
    temp_grad_path = "temp_grad_pdf.png"
    
    try:
        # Save original and gradcam images
        Image.fromarray(original_np).save(temp_orig_path)
        Image.fromarray(gradcam_np).save(temp_grad_path)
        
        # RLImage needs sizes in points. Max width = 250 each to fit side-by-side
        rl_img_orig = RLImage(temp_orig_path, width=245, height=245)
        rl_img_grad = RLImage(temp_grad_path, width=245, height=245)
        
        image_table_data = [
            [Paragraph("<b>Original Leaf Image</b>", ParagraphStyle('CenterBold', parent=bold_body_style, alignment=TA_CENTER)),
             Paragraph("<b>Explainable AI (Grad-CAM Heatmap)</b>", ParagraphStyle('CenterBold', parent=bold_body_style, alignment=TA_CENTER))],
            [rl_img_orig, rl_img_grad]
        ]
        
        image_table = Table(image_table_data, colWidths=[266, 266])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,0), 4),
            ('TOPPADDING', (0,1), (-1,1), 2),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e0e0e0")),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f5f5f5")),
        ]))
        story.append(image_table)
    except Exception as e:
        story.append(Paragraph(f"Error rendering leaf images: {e}", body_style))
    
    story.append(Spacer(1, 15))
    
    # 4. Soil & Weather Tables (Side-by-Side in a multi-column table)
    # Soil Column Table
    soil_rows = [
        [Paragraph("<b>Soil Parameter</b>", bold_body_style), Paragraph("<b>Val</b>", bold_body_style), Paragraph("<b>Status</b>", bold_body_style)],
        [Paragraph("pH", body_style), Paragraph(str(soil_data.get('ph')), body_style), Paragraph("Acidic" if soil_data.get('ph', 6.5) < 5.8 else "Alkaline" if soil_data.get('ph', 6.5) > 7.2 else "Optimal", body_style)],
        [Paragraph("Moisture", body_style), Paragraph(f"{soil_data.get('moisture')}%", body_style), Paragraph("Dry" if soil_data.get('moisture', 50) < 35 else "Wet" if soil_data.get('moisture', 50) > 75 else "Optimal", body_style)],
        [Paragraph("Nitrogen (N)", body_style), Paragraph(f"{soil_data.get('n')}", body_style), Paragraph("Low" if soil_data.get('n', 60) < 40 else "High" if soil_data.get('n', 60) > 120 else "Optimal", body_style)],
        [Paragraph("Phosphorus (P)", body_style), Paragraph(f"{soil_data.get('p')}", body_style), Paragraph("Low" if soil_data.get('p', 50) < 30 else "Optimal", body_style)],
        [Paragraph("Potassium (K)", body_style), Paragraph(f"{soil_data.get('k')}", body_style), Paragraph("Low" if soil_data.get('k', 100) < 70 else "Optimal", body_style)],
    ]
    sub_soil_table = Table(soil_rows, colWidths=[110, 45, 95])
    sub_soil_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e8f5e9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#c8e6c9")),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    # Weather Column Table
    weather_rows = [
        [Paragraph("<b>Weather Metric</b>", bold_body_style), Paragraph("<b>Value</b>", bold_body_style)],
        [Paragraph("Temperature", body_style), Paragraph(f"{weather_data.get('temperature')} °C", body_style)],
        [Paragraph("Humidity", body_style), Paragraph(f"{weather_data.get('humidity')}%", body_style)],
        [Paragraph("Wind Speed", body_style), Paragraph(f"{weather_data.get('wind_speed')} m/s", body_style)],
        [Paragraph("Sky Conditions", body_style), Paragraph(weather_data.get('description', 'N/A').title(), body_style)],
        [Paragraph("Weather Source", body_style), Paragraph("Mock Fallback" if weather_data.get('is_mock') else "Live OpenWeather", body_style)]
    ]
    sub_weather_table = Table(weather_rows, colWidths=[130, 120])
    sub_weather_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e3f2fd")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bbdefb")),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    # Combined side-by-side tables
    combined_table_data = [
        [Paragraph("<b>Soil Diagnostic Data</b>", heading1_style), Paragraph("<b>Environmental Weather Conditions</b>", heading1_style)],
        [sub_soil_table, sub_weather_table]
    ]
    combined_table = Table(combined_table_data, colWidths=[266, 266])
    combined_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,0), 2),
        ('TOPPADDING', (0,1), (-1,1), 2),
    ]))
    
    story.append(combined_table)
    story.append(Spacer(1, 15))
    
    # 5. Approved AI Recommendations Report (Human-in-the-Loop)
    rec_box_content = []
    rec_box_content.append(Paragraph("APPROVED AGRONOMIC RECOMMENDATIONS", heading1_style))
    
    # Convert markdown lines to reportlab paragraphs
    lines = approved_recommendation.split("\n")
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        if line_strip.startswith("###"):
            # Subheading
            heading_text = line_strip.replace("###", "").strip()
            rec_box_content.append(Paragraph(heading_text, rec_heading_style))
        elif line_strip.startswith("-") or line_strip.startswith("*"):
            # Bullet point
            bp_text = line_strip[1:].strip()
            rec_box_content.append(Paragraph(f"&bull; {bp_text}", rec_style))
        else:
            # Paragraph
            rec_box_content.append(Paragraph(line_strip, rec_style))
            
    rec_container_table = Table([[rec_box_content]], colWidths=[532])
    rec_container_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f8e9")),
        ('BOX', (0,0), (-1,-1), 1.5, primary_color),
        ('PADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    
    # Use KeepTogether to ensure recommendation doesn't split awkwardly
    story.append(KeepTogether(rec_container_table))
    story.append(Spacer(1, 15))
    
    # 6. Signature Block
    sig_data = [
        [Paragraph("<b>Farmer / Field Manager Signature:</b> ___________________________", body_style),
         Paragraph("<b>Agronomist Co-Pilot Sign-off:</b> APPROVED", body_style)]
    ]
    sig_table = Table(sig_data, colWidths=[290, 242])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(sig_table)
    
    # Build Document
    doc.build(story)
    
    # Clean up temp image files
    try:
        if os.path.exists(temp_orig_path):
            os.remove(temp_orig_path)
        if os.path.exists(temp_grad_path):
            os.remove(temp_grad_path)
    except Exception as e:
        print(f"Error cleaning up PDF temp images: {e}")
        
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
