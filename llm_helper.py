import os
import openai

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    genai = None
    HAS_GEMINI = False

# Fallback smart recommendation database
DISEASE_INFO = {
    "Corn___Healthy": {
        "pathogen": "None (Healthy Tissue)",
        "explanation": "No symptoms of disease were identified. The leaf tissue shows healthy photosynthetic pigments, cell structures, and vein patterns.",
        "bio": "Maintain soil health by applying compost tea or beneficial mycorrhizal fungi. Implement cover cropping to sustain nutrient retention.",
        "chem": "No chemical treatments or fungicides are recommended. Monitor fields periodically for pest arrivals.",
    },
    "Corn___Common_rust": {
        "pathogen": "Puccinia sorghi (Fungus)",
        "explanation": "Common rust is caused by the fungus Puccinia sorghi. It is characterized by small, powdery, orange-brown pustules (uredinia) that develop on both leaf surfaces. High humidity (relative humidity > 90%) and moderate temperatures (16–23°C) speed up infection.",
        "bio": "Use foliar sprays of Bacillus subtilis or copper soaps. Apply neem oil early in the morning to disrupt spore membrane structure.",
        "chem": "Apply protective triazole (e.g., propiconazole) or strobilurin (e.g., azoxystrobin, pyraclostrobin) fungicides if rust pustules appear on lower leaves before silking.",
    },
    "Potato___Healthy": {
        "pathogen": "None (Healthy Tissue)",
        "explanation": "The potato leaf shows optimal green color and structure. Cell walls and chloroplasts are healthy.",
        "bio": "Employ preventative biological inoculants such as Trichoderma species in the root zone to prevent soil-borne pathogens.",
        "chem": "No chemical treatments are needed. Maintain proper weeding.",
    },
    "Potato___Early_blight": {
        "pathogen": "Alternaria solani (Fungus)",
        "explanation": "Early blight is caused by the fungus Alternaria solani. It targets older foliage first, producing dark brown, circular to angular spots with characteristic 'target' concentric rings. Yellow chlorotic halos often surround these lesions.",
        "bio": "Spray bio-fungicides based on Bacillus amyloliquefaciens. Apply copper hydroxide protectively to limit spore reproduction.",
        "chem": "Spray chlorothalonil, mancozeb, or difenoconazole. Ensure rotational use of FRAC group 7/11 mixtures to prevent resistance buildup.",
    },
    "Potato___Late_blight": {
        "pathogen": "Phytophthora infestans (Oomycete)",
        "explanation": "Late blight is a highly destructive oomycete disease caused by Phytophthora infestans. It produces large, dark, water-soaked lesions on leaves and stems. In damp weather, a white velvety growth of spores appears on the leaf undersides. This disease can defoliate entire fields in days.",
        "bio": "Organic options are limited. Apply heavy protective copper octanoate before rain events. Prune lower canopy to reduce leaf-wetness duration.",
        "chem": "Spray systemic oomycete-specific fungicides like metalaxyl/mefenoxam, dimethomorph, or cyazofamid immediately. Completely remove and incinerate infected plant residues.",
    },
    "Tomato___Healthy": {
        "pathogen": "None (Healthy Tissue)",
        "explanation": "The tomato leaf exhibits balanced chlorophyll levels and normal cell structure, with no signs of fungal or bacterial lesions.",
        "bio": "Inoculate root zone with Glomus intraradices (mycorrhizae) to enhance water and phosphorus uptake efficiency.",
        "chem": "No chemical spraying is recommended. Keep field clean of weeds.",
    },
    "Tomato___Early_blight": {
        "pathogen": "Alternaria solani (Fungus)",
        "explanation": "Early blight on tomatoes is caused by Alternaria solani. It produces brown, concentric-ring spots on lower leaves, stem lesions, and can cause fruit rot. Pruning lower leaves prevents splash-dispersed soil inoculum from reaching the plant.",
        "bio": "Apply Trichoderma harzianum foliar sprays. Apply organic copper fungicide soap at 7-10 day intervals.",
        "chem": "Apply azoxystrobin, chlorothalonil, or copper hydroxide. Rotate chemical groups to maintain treatment efficacy.",
    },
    "Tomato___Late_blight": {
        "pathogen": "Phytophthora infestans (Oomycete)",
        "explanation": "Tomato late blight is caused by Phytophthora infestans. It rapidly infects leaves, stems, and green/red tomato fruits, causing large brown lesions with greasy margins. Spreads rapidly in cool, wet environments.",
        "bio": "Apply preventative copper soaps. Ensure strict sanitation by clearing volunteer nightshade weeds around fields.",
        "chem": "Spray cymoxanil, mandipropamid, or chlorothalonil. Systemic treatments should be applied at the first warning of late blight in the area.",
    }
}

def parse_prediction_label(disease_class):
    """Convert the internal class label into crop and disease strings."""
    parts = disease_class.split('___')
    if len(parts) == 2:
        crop_name = parts[0]
        disease_name = parts[1].replace('_', ' ')
    else:
        crop_name = disease_class
        disease_name = disease_class
    return crop_name, disease_name


def validate_report_matches_prediction(report, crop_name, disease_name):
    normalized = report.lower()
    if crop_name.lower() not in normalized:
        return False
    if disease_name.lower() not in normalized:
        return False
    return True


def generate_cultural_recommendations(soil_data, weather_data):
    """
    Generate customized cultural and physical practices based on soil nutrients and weather conditions.
    """
    recs = []
    
    # Soil pH
    ph = soil_data.get("ph", 6.5)
    if ph < 5.8:
        recs.append(f"Low Soil pH ({ph}): Incorporate agricultural lime (calcium carbonate) or dolomite lime into the soil to raise pH into the optimal 6.0-6.8 range, which increases phosphorus availability.")
    elif ph > 7.2:
        recs.append(f"High Soil pH ({ph}): Apply elemental sulfur or ammonium sulfate fertilizers to lower the pH to the optimal range (6.0-6.8) and avoid iron/zinc chlorosis.")
    else:
        recs.append(f"Optimal Soil pH ({ph}): pH is within the target range for nutrient uptake. Maintain with organic mulch and compost.")
        
    # Soil Moisture
    moisture = soil_data.get("moisture", 50.0)
    if moisture < 35.0:
        recs.append(f"Critical Soil Under-watering ({moisture}%): Implement immediate drip irrigation. Apply a 3-inch layer of straw or organic mulch to suppress evaporation and maintain root coolness.")
    elif moisture > 75.0:
        recs.append(f"Excessive Soil Moisture ({moisture}%): Suspend scheduled irrigation. Clean field drainage ditches to promote surface runoff and improve root zone aeration to prevent damping-off diseases.")
    else:
        recs.append(f"Balanced Soil Moisture ({moisture}%): Continue standard irrigation schedule, targeting watering in the early morning to allow leaves to dry during the day.")
        
    # Nitrogen (N)
    n = soil_data.get("n", 60.0)
    if n < 40.0:
        recs.append(f"Nitrogen Deficiency ({n} mg/kg): Apply a quick-release nitrogen fertilizer (like urea or ammonium nitrate) or organic blood meal to support vegetative vigor.")
    elif n > 120.0:
        recs.append(f"Excessive Nitrogen ({n} mg/kg): High nitrogen promotes lush, soft foliage that is highly susceptible to fungal pathogens like Alternaria and rust. Avoid further nitrogen fertilizers.")
        
    # Phosphorus (P)
    p = soil_data.get("p", 50.0)
    if p < 30.0:
        recs.append(f"Phosphorus Deficiency ({p} mg/kg): Apply bone meal or triple superphosphate (TSP) near the root zone to promote structural root development and flower set.")
        
    # Potassium (K)
    k = soil_data.get("k", 100.0)
    if k < 70.0:
        recs.append(f"Potassium Deficiency ({k} mg/kg): Apply potassium sulfate or muriate of potash. Potassium is critical for cellular water regulation and stomatal resistance, which helps block pathogen entry.")
        
    # Weather-based foliar risk
    temp = weather_data.get("temperature", 22.0)
    humidity = weather_data.get("humidity", 60)
    
    if humidity > 80:
        recs.append(f"High Ambient Humidity Alert ({humidity}%): Leaf wetness duration is elevated, which is high risk for spore germination. Space out crop plants, prune suckers to improve airflow, and restrict overhead watering.")
    if temp > 26.0 and humidity > 70:
        recs.append(f"Warm Humid Conditions ({temp}°C, {humidity}%): Conducive for early blight. Keep close watch for lower leaf spots and apply preventative biological or organic sprays.")
        
    return recs

def get_recommendation_report(disease_class, confidence, soil_data, weather_data, user_desc="", api_key_openai=None, api_key_gemini=None, force_fallback=False):
    """
    Unified LLM call: Generates a complete explanation and recommendations report.
    Checks for Gemini or OpenAI keys. If none are provided, falls back to the smart rule-based generator.
    """
    openai_key = api_key_openai or os.getenv("OPENAI_API_KEY")
    gemini_key = api_key_gemini or os.getenv("GEMINI_API_KEY")
    crop_name, disease_name = parse_prediction_label(disease_class)
    
    crop_info = DISEASE_INFO.get(disease_class, DISEASE_INFO["Corn___Healthy"])
    
    prompt = f"""
You are an expert Agricultural Agronomist and Plant Pathologist.

Predicted Crop: {crop_name}
Predicted Disease: {disease_name}
CNN Confidence Score: {confidence * 100:.1f}%
Pathogen: {crop_info['pathogen']}

Soil Analysis Results:
- pH: {soil_data.get('ph')} (Optimal: 6.0 - 6.8)
- Nitrogen (N): {soil_data.get('n')} mg/kg (Optimal: 40 - 100)
- Phosphorus (P): {soil_data.get('p')} mg/kg (Optimal: 30 - 80)
- Potassium (K): {soil_data.get('k')} mg/kg (Optimal: 70 - 180)
- Soil Moisture: {soil_data.get('moisture')}% (Optimal: 40% - 70%)

Environmental Weather:
- Location/City: {weather_data.get('city')}
- Temperature: {weather_data.get('temperature')}°C
- Humidity: {weather_data.get('humidity')}%
- Wind Speed: {weather_data.get('wind_speed')} m/s
- Sky Conditions: {weather_data.get('description')}

Farmer Observations:
- {user_desc if user_desc else 'No notes provided.'}

Instructions:
- Use ONLY the predicted crop and predicted disease in this report.
- Do NOT change the predicted crop ({crop_name}) or the predicted disease ({disease_name}).
- Do NOT mention or recommend any disease other than the detected disease.
- Do NOT include example diseases or alternate diagnoses.
- If the confidence is low, explicitly mention uncertainty but do not substitute a different diagnosis.
- Provide recommendations only for the detected disease and its mapped pathogen.

Please structure your report in markdown with the following sections:
### 1. Disease Pathology & Explanation
### 2. Treatment Strategy: Biological Controls
### 3. Treatment Strategy: Chemical Controls
### 4. Treatment Strategy: Cultural & Soil Amendments
"""

    # 2. Try Gemini API if not forcing fallback
    if gemini_key and not force_fallback:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            if response.text and validate_report_matches_prediction(response.text, crop_name, disease_name):
                return response.text
            elif response.text:
                print("Gemini response failed validation. Falling back to rule-based report.")
        except Exception as e:
            print(f"Gemini API execution failed: {e}. Trying OpenAI...")
            
    # 3. Try OpenAI API if not forcing fallback
    if openai_key and not force_fallback:
        try:
            client = openai.OpenAI(api_key=openai_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional agricultural advisor. Provide precise, structured agronomic recommendations."},
                    {"role": "user", "content": prompt}
                ]
            )
            if completion.choices[0].message.content:
                response_text = completion.choices[0].message.content
                if validate_report_matches_prediction(response_text, crop_name, disease_name):
                    return response_text
                print("OpenAI response failed validation. Falling back to rule-based report.")
        except Exception as e:
            print(f"OpenAI API execution failed: {e}. Falling back to Rule-Based system...")

    # 4. RULE-BASED FALLBACK GENERATOR (Deterministic, extremely detailed and specific)
    cultural_recs = generate_cultural_recommendations(soil_data, weather_data)
    cultural_str = "\n".join([f"- {rec}" for rec in cultural_recs])
    
    fallback_report = f"""### 1. Disease Pathology & Explanation
- **Crop**: {crop_name}
- **Disease Target**: {disease_name} (CNN Confidence: {confidence * 100:.1f}%)
- **Pathogen Agent**: *{crop_info['pathogen']}*
- **Aetiology**: {crop_info['explanation']}
- **Weather Factor**: Current ambient temperature of {weather_data.get('temperature')}°C and humidity of {weather_data.get('humidity')}% are playing a key role in the crop microclimate. High relative humidity increases the spore hydration levels required for foliar penetration.

### 2. Treatment Strategy: Biological Controls
- **Organic Inoculants**: {crop_info['bio']}
- **Foliar Wash**: Apply neem oil or horticultural soaps to form a protective layer on remaining leaves. Ensure application occurs during late afternoon to prevent sun-induced leaf burn.

### 3. Treatment Strategy: Chemical Controls
- **Fungicide Spray**: {crop_info['chem']}
- **FRAC Code Rotation**: Rotate fungicide classes according to label recommendations to prevent resistance buildup. Always respect harvest intervals (PHI).

### 4. Treatment Strategy: Cultural & Soil Amendments
{cultural_str}
- **Foliage Spacing**: Maintain adequate plant spacing to promote healthy wind circulation, keeping leaves dry and preventing spores from spreading.
- **Sanitation**: Clean tools between plots and remove infected leaves immediately.
"""
    return fallback_report
