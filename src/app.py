import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.classifier import create_swin_dr_classifier
from src.utils.risk_calculator import RiskCalculator
from src.explainability.heatmaps import ExplainabilityEngine
from src.preprocessing.transforms import BenGrahamPreprocessingD
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord

# Page Config
st.set_page_config(
    page_title="Diabetic Retinopathy Clinical Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DR_LEVELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

@st.cache_resource
def load_model():
    """Load the 3-Model Ensemble (Swin + EffNet + DenseNet)."""
    from src.models.ensemble_factory import MetaPredictor
    
    # Path to our best Swin-Tiny model (Phase 3 adapted)
    swin_path = "production/weights/production_v1.pth"
    # New fine-tuned auxiliary models (Phase 8 Balanced)
    effnet_path = "production/weights/efficientnet_b0_balanced.pth"
    densenet_path = "production/weights/densenet121_aptos_finetuned.pth"
    
    # Initialize MetaPredictor (Ensemble)
    # Check if files exist to avoid crash if training isn't done yet (though it should be)
    kwargs = {'swin_path': swin_path}
    if os.path.exists(effnet_path):
        kwargs['effnet_path'] = effnet_path
    if os.path.exists(densenet_path):
        kwargs['densenet_path'] = densenet_path
        
    predictor = MetaPredictor(**kwargs)
    
    # Initialize Explainability Engine (using the Swin model from the predictor for now)
    swin_model = predictor._load_swin()
    explainer = ExplainabilityEngine(swin_model)
    
    return predictor, explainer, predictor.device, 1.0

def get_transforms():
    """Get inference transforms."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(224, 224)),
        # Note: Scaling moved after BenGraham in pipeline fix
        BenGrahamPreprocessingD(keys=["image"], sigma=10),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ])

def main():
    st.title("👁️ Diabetic Retinopathy Clinical Assistant")
    st.markdown("### AI-Powered Screening & Risk Assessment")
    
    # Sidebar: Patient Info
    st.sidebar.header("Patient Data")
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=55)
    hba1c = st.sidebar.slider("HbA1c (%)", 4.0, 15.0, 6.5, 0.1)
    duration = st.sidebar.slider("Diabetes Duration (Years)", 0, 50, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Image Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])
    
    # Load Model
    with st.spinner("Loading Medical AI Engine..."):
        model, explainer, device, temperature = load_model()
    st.sidebar.success("System Ready")

    if uploaded_file is not None:
        # Create temp file for MONAI loader
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_file.getvalue())
        tfile.close()
        
        # Display Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Fundus Image")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_container_width=True)

        # Process Image
        with st.spinner("Analyzing Retinal Structure..."):
            transforms = get_transforms()
            data = {"image": tfile.name}
            data = transforms(data)
            input_tensor = data["image"].unsqueeze(0).to(device)
            
            # Inference (Sequential Ensemble)
            # Note: predict_sequential handles to(device) internally for the model
            input_tensor = data["image"].unsqueeze(0)
            final_probs, breakdown = model.predict_sequential(input_tensor)
            
            # Helper to get prediction details
            prediction = breakdown['Ensemble']
            confidence = breakdown.get('Confidence', torch.max(final_probs).item())
            prob_referable = final_probs[0][2:].sum().item()
            
            # --- ENSEMBLE VOTING BREAKDOWN ---
            st.divider()
            st.subheader("🤖 AI Council Consensus")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Swin-Tiny (Structure)", f"Grade {breakdown['Swin']}", help="Weight: 0.4")
            c2.metric("EfficientNet (Severity)", f"Grade {breakdown['EfficientNet']}", help="Weight: 0.3")
            c3.metric("DenseNet (Early DR)", f"Grade {breakdown['DenseNet']}", help="Weight: 0.3")
            
            # Color code final decision
            final_grade = breakdown['Ensemble']
            final_color = "normal" if final_grade == 0 else "inverse"
            c4.metric("🏆 Final Decision", f"Grade {final_grade}", delta_color=final_color)
            
            # Safety Alert for Ambiguous Cases
            if breakdown.get('Ambiguous_Flag', False):
                st.error("⚠️ **AMBIGUOUS CASE: Manual Clinical Review Required**")
                st.caption(f"The Ensemble Confidence is within the Grey-Zone ({breakdown['Confidence']:.1%}). This image may contain atypical features or be positioned between typical DR grades.")
            st.divider()
            # ---------------------------------
            
            # Explainability (Heatmap)
            # Re-load Swin for explanation (since it was unloaded)
            # This is a bit inefficient but safe for VRAM
            swin_model = model._load_swin()
            explainer.model = swin_model # Update explainer model
            heatmap = explainer.generate_heatmap(input_tensor, target_class=prediction)
            del swin_model
            torch.cuda.empty_cache()
            
            # Overlay Heatmap
            orig_cv = np.array(original_image.convert('RGB'))
            overlay = ExplainabilityEngine.overlay_heatmap(orig_cv, heatmap, alpha=0.4)
            
            # Risk Calculation (Clinical Heuristic)
            clinical_risk = RiskCalculator.calculate_risk(prediction, hba1c, duration)
        
        with col2:
            st.subheader("AI Lesion Detection (Heatmap)")
            st.image(overlay, use_container_width=True, caption="Warm regions indicate areas influential to diagnosis")
            
        # Cleanup
        os.unlink(tfile.name)
        
        # Counseling Report Section
        st.markdown("---")
        st.header("📄 Clinical Counseling Report")
        
        # Grading Result
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted DR Grade", f"{prediction} - {DR_LEVELS[prediction]}", f"{confidence:.1%}")
        
        # Safety Filter (Ambiguity Check)
        if 0.4 <= prob_referable <= 0.6:
            c2.warning("⚠️ Result Ambiguous")
            st.warning(f"**Safety Alert**: The model is uncertain (Referable Probability: {prob_referable:.2f}). **Doctor Verification Required.**")
        else:
            c2.metric("Referable DR Probability", f"{prob_referable:.1%}", delta_color="inverse" if prob_referable > 0.5 else "normal")
            
        c3.metric("Clinical Risk Index", f"{clinical_risk['score']}", delta_color="inverse" if clinical_risk["score"] > 8 else "normal")
        
        # Detailed Advice
        st.info(f"**Recommendation**: {clinical_risk['recommendation']}")
        
        with st.expander("Detailed Risk Factors"):
            st.write(f"**HbA1c**: {hba1c}%")
            st.write(f"**Duration**: {duration} years")
            st.write(f"**DR Grade**: {DR_LEVELS[prediction]}")
            st.write(f"**AI Confidence**: {confidence:.1%}")
            st.write(f"**Calibration Temp**: {temperature:.3f}")
            st.caption("Clinical Risk Index Formula: (Grade × 0.5) + (HbA1c × 0.3) + (Duration × 0.2)")
        
        # HITL Feedback Section
        st.markdown("---")
        st.header("🩺 Doctor's Feedback (Human-in-the-Loop)")
        st.markdown("*Help improve the AI by providing corrections on misdiagnosed cases*")
        
        col_feedback1, col_feedback2 = st.columns([1, 2])
        
        with col_feedback1:
            st.subheader("Correction")
            correct_grade = st.selectbox(
                "Correct Grade (if AI is wrong)",
                options=list(DR_LEVELS.keys()),
                format_func=lambda x: f"{x} - {DR_LEVELS[x]}",
                index=prediction,
                key="doctor_grade"
            )
            
        with col_feedback2:
            st.subheader("Clinical Notes")
            clinical_notes = st.text_area(
                "Doctor's Clinical Observations",
                placeholder="e.g., 'Microaneurysms visible in superior temporal quadrant, AI missed early signs...'",
                height=100,
                key="clinical_notes"
            )
        
        if st.button("📝 Submit Clinical Review", type="primary"):
            import csv
            from datetime import datetime
            
            feedback_csv = Path("data/feedback/doctor_corrections.csv")
            feedback_csv.parent.mkdir(parents=True, exist_ok=True)
            
            # Save feedback
            with open(feedback_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    uploaded_file.name,
                    prediction,
                    correct_grade,
                    f"{confidence:.4f}",
                    clinical_notes.replace('\n', ' ')
                ])
            
            st.success(f"✅ Clinical Review Saved! AI predicted: {DR_LEVELS[prediction]}, Doctor corrected to: {DR_LEVELS[correct_grade]}")
            st.info("💡 Tip: Run `python -m src.train_feedback` to fine-tune the model on corrected cases")

if __name__ == "__main__":
    main()
