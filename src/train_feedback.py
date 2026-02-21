# ... imports ...
from src.data.loader import DRMedicalLoader
import random

# ... CONFIG ...
CONFIG = {
    "feedback_csv": "data/feedback/doctor_corrections.csv",
    "model_checkpoint": "experiments/best_model.pth",
    "output_model": "experiments/feedback_tuned_model.pth",
    "epochs": 5, 
    "batch_size": 4, 
    "lr": 1e-6, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "regression_threshold": 0.85 # Minimum accuracy on standard set to accept new model
}

# ... FeedbackDataset class ...

def freeze_backbone(model):
    """
    Freeze all layers except the last Swin Block and the Head.
    SwinTiny has 4 stages. We freeze stages 0, 1, 2.
    Keep stage 3 (last block) and classifier trainable.
    """
    print("❄️ Freezing early layers...")
    
    # Freeze entire backbone first
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Unfreeze the last stage (Stage 3) of the Swin Encoder
    # Monai SwinUNETR structure: model.backbone.swinViT.layers is a ModuleList
    # Standard Swin-T has 4 layers.
    last_layer = model.backbone.swinViT.layers[-1]
    for param in last_layer.parameters():
        param.requires_grad = True
        
    # Unfreeze Norm layers if necessary (often good practice to keep norms trainable or frozen? 
    # Usually keep frozen for few-shot to avoid stat shift. Monai Swin norms are inside layers usually.)
    # We specifically ensure Classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Verify Trainable Parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable Params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({(trainable/total)*100:.1f}%)")


class RegressionValidator:
    """Checks if the new model has degraded on standard cases."""
    def __init__(self, device):
        self.device = device
        self.loader = self._get_standard_validation_set()
        
    def _get_standard_validation_set(self):
        # Load standard validation set from DRMedicalLoader
        # We need about 50 images.
        print("📊 Loading regression safety set...")
        loader_obj = DRMedicalLoader(
            data_dir="data/raw/ddr", 
            csv_file="data/raw/ddr/DR_grading.csv",
            batch_size=16
        )
        _, val_loader = loader_obj.get_loaders()
        return val_loader

    def validate(self, model):
        model.eval()
        correct = 0
        total = 0
        limit = 50 # Check 50 cases
        
        with torch.no_grad():
            for images, labels in self.loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if total >= limit:
                    break
        
        acc = correct / total
        print(f"🛡️ Safety Check (Standard Data): Acc = {acc:.2%}")
        return acc

def train_feedback():
    print("=" * 60)
    print("HITL Few-Shot Fine-Tuning (Adaptive)")
    print("=" * 60)
    
    # ... Feedback Checks (keep existing code) ...
    # (Simplified for brevity in replacement, but logically we need to check CSV existence again)
    feedback_path = Path(CONFIG["feedback_csv"])
    if not feedback_path.exists():
        print("❌ No feedback file found.")
        return
        
    df = pd.read_csv(feedback_path)
    dataset = FeedbackDataset(CONFIG["feedback_csv"], transform=Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(224, 224)),
        BenGrahamPreprocessingD(keys=["image"], sigma=10),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"])
    ]))
    
    if len(dataset) == 0:
        print("❌ No valid images.")
        return

    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # Load Model
    print("Loading base model...")
    model = create_swin_dr_classifier(vram_optimized=True)
    checkpoint_path = Path(CONFIG["model_checkpoint"])
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG["device"]))
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    
    model = model.to(CONFIG["device"])
    
    # 1. Freeze Layers
    freeze_backbone(model)
    
    # Optimizer (Micro-learning rate)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=CONFIG["lr"], weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Fine-tuning Loop
    mlflow.set_experiment("HITL_Feedback_Tuning")
    with mlflow.start_run():
        mlflow.log_params(CONFIG)
        
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(loader, desc=f"Fine-Tuning Epoch {epoch+1}")
            for images, labels in pbar:
                images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({"Loss": loss.item()})

            avg_loss = total_loss / len(loader)
            acc = correct / total
            print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Acc {acc:.2%}")

    # 2. Regression Safety Check
    print("\n🔍 Running Performance Regression Check...")
    validator = RegressionValidator(CONFIG["device"])
    safety_acc = validator.validate(model)
    
    if safety_acc >= CONFIG["regression_threshold"]:
        print("✅ Safety Check PASSED. Model is stable.")
        torch.save(model.state_dict(), CONFIG["output_model"])
        print(f"💾 Saved Fine-Tuned Model: {CONFIG['output_model']}")
    else:
        print(f"⚠️ Safety Check FAILED ({safety_acc:.2%} < {CONFIG['regression_threshold']:.2%})")
        print("🛑 Model degraded on standard cases. Discarding updates.")
        # We do NOT save the model

if __name__ == "__main__":
    train_feedback()
