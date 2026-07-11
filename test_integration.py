import os
import numpy as np
from PIL import Image

def test_full_pipeline():
    print("==================================================")
    print("RUNNING AGRICULTURE ADVISOR PIPELINE INTEGRATION TEST")
    print("==================================================")
    
    # 1. Verify files exist
    assert os.path.exists("plant_disease_model.keras"), "CNN model file not found!"
    assert os.path.exists("samples/corn_common_rust.jpg"), "Sample leaf image not found!"
    print("[PASS] Model and sample directories verified.")
    
    # 2. Test Model Inference & Grad-CAM
    import model
    print("Running leaf disease diagnosis test on Corn Common Rust...")
    results = model.predict_leaf_disease("samples/corn_common_rust.jpg")
    
    assert results["class_name"] == "Corn___Common_rust", f"Expected Corn___Common_rust but got {results['class_name']}"
    assert results["confidence"] > 0.8, f"Expected high confidence but got {results['confidence']}"
    assert results["original_img"].shape == (224, 224, 3), "Original image shape mismatch"
    assert results["overlaid_img"].shape == (224, 224, 3), "Overlaid image shape mismatch"
    assert results["heatmap"].shape == (224, 224, 3), "Heatmap shape mismatch"
    
    print(f"[PASS] Disease prediction success: {results['display_name']} (Conf: {results['confidence']:.2f})")
    print("[PASS] Grad-CAM heatmap and overlays generated successfully.")
    
    # 3. Test Weather helper
    import weather_helper
    print("Running weather fetching test (Miami)...")
    weather = weather_helper.fetch_weather("Miami")
    assert weather["city"] == "Miami", "City name mismatch"
    assert "temperature" in weather, "Temp missing in weather"
    assert "humidity" in weather, "Humidity missing in weather"
    print(f"[PASS] Weather fetch success: {weather['temperature']} C, humidity {weather['humidity']}%, desc: {weather['description']}")
    
    # 4. Test LLM helper (with rule-based fallback)
    import llm_helper
    print("Running LLM advisor recommendation compiler...")
    soil_data = {"ph": 6.2, "n": 30, "p": 25, "k": 110, "moisture": 32}
    
    report = llm_helper.get_recommendation_report(
        disease_class=results["class_name"],
        confidence=results["confidence"],
        soil_data=soil_data,
        weather_data=weather,
        user_desc="I see reddish dust on my crop leaves in rows 2-4."
    )
    
    assert "### 1. Disease Pathology" in report, "Report missing explanation heading"
    assert "### 4. Treatment Strategy: Cultural" in report, "Report missing cultural amendments heading"
    # Verify soil dynamic edits are in the report
    assert "Low Soil Moisture" in report or "Soil Under-watering" in report or "moisture" in report.lower(), "Soil moisture advice missing"
    assert "Nitrogen Deficiency" in report or "nitrogen" in report.lower(), "Nitrogen deficiency advice missing"
    print("[PASS] LLM advisor report successfully generated with dynamic soil-agronomic context.")
    
    # 5. Test PDF generation compiles successfully
    import report_helper
    print("Compiling PDF report...")
    pdf_bytes = report_helper.generate_pdf_report(
        diagnosis_name=results["display_name"],
        confidence=results["confidence"],
        severity=results["severity"],
        severity_color=results["severity_color"],
        original_np=results["original_img"],
        gradcam_np=results["overlaid_img"],
        soil_data=soil_data,
        weather_data=weather,
        approved_recommendation=report
    )
    
    assert len(pdf_bytes) > 0, "Generated PDF is empty!"
    
    # Write to a test report file to verify it visually
    test_pdf_path = "test_report.pdf"
    with open(test_pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    assert os.path.exists(test_pdf_path), "Failed to save test PDF report to disk!"
    assert os.path.getsize(test_pdf_path) > 1000, "Saved test PDF report is suspiciously small!"
    print(f"[PASS] PDF report compiled and saved successfully to {test_pdf_path} (Size: {os.path.getsize(test_pdf_path)} bytes)")
    
    # Clean up test report file
    if os.path.exists(test_pdf_path):
        os.remove(test_pdf_path)
        print("[PASS] Temporary test PDF report cleaned up.")
        
    print("\n==================================================")
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_full_pipeline()
