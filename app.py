import os
import streamlit as st
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Import project helper modules
import model
import weather_helper
import llm_helper
import report_helper

# 1. Page Configuration
st.set_page_config(
    page_title="Agriculture Advisor AI Co-Pilot",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Premium Custom CSS for Visual Excellence
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Apply modern typography */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Sidebar Custom Styling */
.css-1542fc6, [data-testid="stSidebar"] {
    background-color: #1e3f20;
    color: #ffffff;
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
    color: #f1f8e9 !important;
}

/* Premium Card Panels */
.card-panel {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
    border-left: 5px solid #2e7d32;
}

.weather-card {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 5px solid #1976d2;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}

.soil-card {
    background: linear-gradient(135deg, #efebe9 0%, #d7ccc8 100%);
    border-left: 5px solid #5d4037;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}

/* Title banner */
.title-banner {
    background: linear-gradient(90deg, #1b5e20 0%, #33691e 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(27, 94, 32, 0.2);
}

.title-banner h1 {
    margin: 0;
    font-weight: 700;
    font-size: 2.2rem;
    color: #ffffff !important;
}
.title-banner p {
    margin: 8px 0 0 0;
    font-weight: 300;
    font-size: 1.1rem;
    color: #e8f5e9 !important;
}

/* Status Badges */
.status-badge {
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}
.status-draft {
    background-color: #ffe082;
    color: #8f6a00;
}
.status-approved {
    background-color: #c8e6c9;
    color: #1b5e20;
}
.status-rejected {
    background-color: #ffcdd2;
    color: #b71c1c;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Session State Initialization
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "weather" not in st.session_state:
    st.session_state.weather = None
if "rec_text" not in st.session_state:
    st.session_state.rec_text = ""
if "rec_original" not in st.session_state:
    st.session_state.rec_original = ""
if "rec_status" not in st.session_state:
    st.session_state.rec_status = "Draft"  # Draft, Approved, Rejected

# 4. SIDEBAR - CONTROL PANEL
st.sidebar.markdown("<h2 style='text-align: center; color: white;'>⚙️ Advisor Co-Pilot</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Section: API Configurations (Collapsible)
with st.sidebar.expander("🔑 API Credentials (Optional)", expanded=False):
    st.markdown("<small>Provide API keys to activate live data. If empty, the app runs offline with detailed synthetic models.</small>", unsafe_allow_html=True)
    openai_key = st.text_input("OpenAI API Key", type="password", value="")
    gemini_key = st.text_input("Gemini API Key", type="password", value="")
    weather_key = st.text_input("OpenWeather API Key", type="password", value="")

# Section: Modality 1 - Leaf Image Upload / Select
st.sidebar.markdown("### 🍃 1. Leaf Image Input")
image_file = st.sidebar.file_uploader("Upload Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Quick select samples for quick hackathon demonstrations
st.sidebar.markdown("<p style='margin-bottom:0px;'><small><b>Or Quick Test with Sample Leaf:</b></small></p>", unsafe_allow_html=True)
sample_options = ["None"] + [name.replace("___", " - ").replace("_", " ") for name in model.CLASSES]
sample_select = st.sidebar.selectbox("Choose a pre-generated sample", sample_options)

selected_image = None
if image_file is not None:
    selected_image = Image.open(image_file)
    st.sidebar.success("Uploaded image loaded.")
elif sample_select != "None":
    # Load the corresponding sample file
    formatted_name = sample_select.lower().replace(" - ", "_").replace(" ", "_") + ".jpg"
    sample_path = os.path.join("samples", formatted_name)
    if os.path.exists(sample_path):
        selected_image = Image.open(sample_path)
        st.sidebar.info(f"Loaded sample: {sample_select}")
    else:
        st.sidebar.error(f"Sample file {sample_path} not found. Run train_model.py.")

st.sidebar.markdown("---")

# Section: Modality 2 - Soil Values
st.sidebar.markdown("### 🧪 2. Soil Analysis Data")
soil_ph = st.sidebar.slider("Soil pH (Acidity)", 4.0, 9.0, 6.5, step=0.1)
soil_n = st.sidebar.slider("Nitrogen (N) - mg/kg", 0, 200, 65)
soil_p = st.sidebar.slider("Phosphorus (P) - mg/kg", 0, 150, 48)
soil_k = st.sidebar.slider("Potassium (K) - mg/kg", 0, 250, 110)
soil_moisture = st.sidebar.slider("Soil Moisture Content (%)", 10, 100, 52)

soil_dict = {"ph": soil_ph, "n": soil_n, "p": soil_p, "k": soil_k, "moisture": soil_moisture}

# Section: Modality 3 - Text Observations
st.sidebar.markdown("### 📝 3. Field Observations")
user_description = st.sidebar.text_area("Describe plant symptoms, field conditions, or history:", placeholder="e.g. Yellow spots on leaves, high weed density around crop row 3...")

# Weather City Location
st.sidebar.markdown("### 📍 4. Location")
city_location = st.sidebar.text_input("Enter City / Farm Location", value="Des Moines")

st.sidebar.markdown("---")

# Predict Button
run_predict = st.sidebar.button("🚀 Analyze & Generate Report", use_container_width=True)

# 5. MAIN PAGE - DASHBOARD HEADER
st.markdown(
    """
    <div class="title-banner">
        <h1>Agriculture Advisor AI Co-Pilot</h1>
        <p>Expert Agronomic Diagnostics, Explainable AI Heatmaps & Soil/Weather Advisory</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 6. TRIGGER PREDICTION WORKFLOW
if run_predict:
    if selected_image is None:
        st.warning("⚠️ Please upload a leaf image or select a sample leaf image in the sidebar first.")
    else:
        with st.spinner("Processing modalities (Leaf image CNN, Soil parameters, Weather API)..."):
            # 1. Fetch Weather
            weather_data = weather_helper.fetch_weather(city_location, api_key=weather_key)
            st.session_state.weather = weather_data
            
            # 2. Run CNN Predictor & Grad-CAM
            # Save temporary image for prediction module
            temp_path = "temp_predict_input.jpg"
            selected_image.save(temp_path)
            diag_results = model.predict_leaf_disease(temp_path)
            st.session_state.prediction_results = diag_results
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # 3. Query LLM Recommendations
            ai_rec = llm_helper.get_recommendation_report(
                disease_class=diag_results["class_name"],
                confidence=diag_results["confidence"],
                soil_data=soil_dict,
                weather_data=weather_data,
                user_desc=user_description,
                api_key_openai=openai_key,
                api_key_gemini=gemini_key
            )
            st.session_state.rec_original = ai_rec
            st.session_state.rec_text = ai_rec
            st.session_state.rec_status = "Draft"  # Reset status to draft for review
            st.session_state.predict_clicked = True
            
            st.success("✅ Analysis completed successfully!")

# 7. DISPLAY DASHBOARD CONTENTS
if st.session_state.predict_clicked and st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results
    weather = st.session_state.weather
    
    # Create Tabs for neat structure
    tab_diag, tab_rec, tab_soil, tab_export = st.tabs([
        "🔍 Tab 1: AI Diagnosis & Explainable AI", 
        "🤖 Tab 2: Co-Pilot Recommendations (Human-in-the-Loop)", 
        "📊 Tab 3: Soil Health Analytics",
        "📄 Tab 4: Export PDF Report"
    ])
    
    # --- TAB 1: DIAGNOSIS & EXPLAINABLE AI ---
    with tab_diag:
        st.markdown("### Diagnostic Analysis")
        
        # Grid layout for Leaf diagnosis summary
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(
                f"""
                <div class="card-panel">
                    <h4>Identified Plant Pathogen</h4>
                    <h2 style='color: {results["severity_color"]}; margin-top: 0px;'>{results["display_name"]}</h2>
                    <p>Severity Level: <b><font color='{results["severity_color"]}'>{results["severity"]}</font></b></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            # Confidence Card
            pct_conf = int(results["confidence"] * 100)
            st.markdown(
                f"""
                <div class="card-panel" style="text-align: center; border-left-color: #ff9800;">
                    <h4>CNN Confidence</h4>
                    <h1 style='color: #ff9800; font-size: 3.5rem; margin: 0;'>{pct_conf}%</h1>
                    <p style='color: #757575;'>Model Probability</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col3:
            # Local Weather Card
            st.markdown(
                f"""
                <div class="weather-card">
                    <h4>📍 {weather["city"]}</h4>
                    <h2 style='margin: 0; color: #1976d2;'>{weather["temperature"]}°C</h2>
                    <p style='margin: 4px 0 0 0;'><b>Humidity:</b> {weather["humidity"]}%</p>
                    <p style='margin: 2px 0 0 0;'><b>Wind:</b> {weather["wind_speed"]} m/s</p>
                    <p style='margin: 2px 0 0 0; text-transform: capitalize;'><b>Condition:</b> {weather["description"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("---")
        
        # Image side-by-side comparison
        st.markdown("### Explainable AI: Grad-CAM Saliency Visualizations")
        img_col1, img_col2, img_col3 = st.columns(3)
        
        with img_col1:
            st.image(results["original_img"], caption="Original Uploaded Leaf Image", use_container_width=True)
        with img_col2:
            st.image(results["heatmap"], caption="Grad-CAM Saliency Heatmap (JET)", use_container_width=True)
        with img_col3:
            st.image(results["overlaid_img"], caption="Overlaid Activation (Conv_2 Layer)", use_container_width=True)
            
        st.markdown(
            """
            > **How to interpret Grad-CAM:** The red/orange zones highlight the spatial areas of the leaf that 
            contributed most heavily to the CNN's classification score. This shows that the neural network is 
            legitimately looking at the disease lesions/pustules rather than the background desk or leaves.
            """
        )

    # --- TAB 2: RECOMMENDATIONS & HUMAN-IN-THE-LOOP ---
    with tab_rec:
        st.markdown("### AI Co-Pilot Recommendation Report")
        
        # Display Current Approval Status
        status_class = "status-draft"
        if st.session_state.rec_status == "Approved":
            status_class = "status-approved"
        elif st.session_state.rec_status == "Rejected":
            status_class = "status-rejected"
            
        st.markdown(
            f"""
            <div style='margin-bottom: 15px;'>
                <span><b>Human-in-the-Loop Status:</b></span>
                <span class="status-badge {status_class}">{st.session_state.rec_status.upper()}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Editable Recommendations Text Area
        edited_text = st.text_area(
            "Review, edit, and approve the recommended treatment strategy:",
            value=st.session_state.rec_text,
            height=400,
            key="rec_text_area"
        )
        # Update session state with edits
        st.session_state.rec_text = edited_text
        
        # Actions Row
        col_btn1, col_btn2, col_btn3, _ = st.columns([1.2, 1.2, 1.2, 3])
        
        with col_btn1:
            if st.button("✅ Approve Recommendation", use_container_width=True, type="primary"):
                st.session_state.rec_status = "Approved"
                st.rerun()
        with col_btn2:
            if st.button("❌ Reject Recommendation", use_container_width=True):
                st.session_state.rec_status = "Rejected"
                st.rerun()
        with col_btn3:
            if st.button("🔄 Reset to AI Original", use_container_width=True):
                st.session_state.rec_text = st.session_state.rec_original
                st.session_state.rec_status = "Draft"
                st.rerun()

    # --- TAB 3: SOIL HEALTH ANALYTICS ---
    with tab_soil:
        st.markdown("### Soil Health Assessment")
        
        col_soil1, col_soil2 = st.columns([3, 2])
        
        with col_soil1:
            st.markdown("#### Nutrients & Parameters vs Optimal Ranges")
            
            # Show parameters as meters/gauges
            def get_status_text(val, opt_min, opt_max):
                if val < opt_min:
                    return f"<span style='color: red;'><b>Deficient</b> ({val} vs target {opt_min}-{opt_max})</span>"
                elif val > opt_max:
                    return f"<span style='color: red;'><b>Excessive</b> ({val} vs target {opt_min}-{opt_max})</span>"
                else:
                    return f"<span style='color: green;'><b>Optimal</b> ({val} vs target {opt_min}-{opt_max})</span>"

            st.markdown(
                f"""
                <div class="soil-card">
                    <p style='margin:0px;'><b>Nitrogen (N):</b> {get_status_text(soil_n, 40, 100)}</p>
                    <p style='margin:5px 0px 0px 0px;'><b>Phosphorus (P):</b> {get_status_text(soil_p, 30, 80)}</p>
                    <p style='margin:5px 0px 0px 0px;'><b>Potassium (K):</b> {get_status_text(soil_k, 70, 180)}</p>
                    <p style='margin:5px 0px 0px 0px;'><b>Soil pH:</b> {get_status_text(soil_ph, 6.0, 6.8)}</p>
                    <p style='margin:5px 0px 0px 0px;'><b>Soil Moisture:</b> {get_status_text(soil_moisture, 40, 70)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_soil2:
            st.markdown("#### Cultural Agronomic Actions")
            cultural_recs = llm_helper.generate_cultural_recommendations(soil_dict, weather)
            for rec in cultural_recs:
                st.markdown(f"- {rec}")

    # --- TAB 4: EXPORT REPORT ---
    with tab_export:
        st.markdown("### Compile and Download PDF Report")
        
        if st.session_state.rec_status != "Approved":
            st.warning("⚠️ **Human-in-the-Loop Safeguard Active:** You must review and **Approve** the recommendations in Tab 2 before the PDF report can be compiled and downloaded.")
        else:
            st.success("🎉 **Recommendation Approved!** The PDF report is ready for export.")
            
            # Generate PDF bytes
            pdf_bytes = report_helper.generate_pdf_report(
                diagnosis_name=results["display_name"],
                confidence=results["confidence"],
                severity=results["severity"],
                severity_color=results["severity_color"],
                original_np=results["original_img"],
                gradcam_np=results["overlaid_img"],
                soil_data=soil_dict,
                weather_data=weather,
                approved_recommendation=st.session_state.rec_text
            )
            
            # Download Button
            st.download_button(
                label="📥 Download Diagnostic PDF Report",
                data=pdf_bytes,
                file_name=f"agronomic_report_{city_location.lower().replace(' ', '_')}_{results['class_name'].lower().replace('___', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            st.info("The generated report includes the side-by-side Grad-CAM heatmap visualizations, localized weather stats, chemical and biological controls, soil deficiency amendments, and your signed approval block.")

else:
    # Landing Screen (Before Run Diagnosis)
    st.info("👋 **Welcome to the Agriculture Advisor AI Co-Pilot!** Get started by configuring your farm details:")
    st.markdown(
        """
        1. **Select or upload a leaf image** in the sidebar. We suggest starting with one of the pre-made sample leaves (like *Corn Rust* or *Potato Early Blight*) to see the CNN classifier and Grad-CAM explainable AI in action immediately.
        2. **Enter the soil chemical analysis values** (pH, Nitrogen, Phosphorus, Potassium, Moisture).
        3. **Provide field notes** of what you observe in the crops.
        4. **Define your location** for local weather checks.
        5. **Click 'Analyze & Generate Report'** in the sidebar to process the data!
        """
    )
    
    # Visual grid of sample crops
    st.markdown("### Sample Leaves Available for Fast Demonstration")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    with sample_col1:
        st.markdown("**Corn Samples**")
        st.write("- Healthy Leaf")
        st.write("- Common Rust (Orange spots)")
    with sample_col2:
        st.markdown("**Potato Samples**")
        st.write("- Healthy Leaf")
        st.write("- Early Blight (Concentric circles)")
        st.write("- Late Blight (Water patches)")
    with sample_col3:
        st.markdown("**Tomato Samples**")
        st.write("- Healthy Leaf")
        st.write("- Early Blight (Concentric circles)")
        st.write("- Late Blight (Water patches)")
    with sample_col4:
        st.image("samples/corn_common_rust.jpg", caption="Generated Sample: Corn Rust Leaf", use_container_width=True)
