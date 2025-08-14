"""
Run Fixed Slum Detection System
==============================
"""

def run_fixed_system():
    print("🔧 RUNNING FIXED SLUM DETECTION SYSTEM")
    print("=" * 50)
    
    # Step 1: Fixed dataset analysis
    print("\n📊 Step 1: Fixed dataset analysis...")
    try:
        from fixed_dataset_analyzer import FixedSlumDatasetAnalyzer
        analyzer = FixedSlumDatasetAnalyzer()
        results = analyzer.analyze_complete_dataset()
        print("✅ Fixed analysis complete")
    except Exception as e:
        print(f"⚠️ Analysis failed: {e}")
    
    # Step 2: Train fixed model
    print("\n🏋️ Step 2: Training fixed model...")
    try:
        from fixed_trainer import FixedTrainer
        trainer = FixedTrainer()
        best_loss = trainer.train(epochs=20, batch_size=4)
        print(f"✅ Training complete! Best loss: {best_loss:.4f}")
    except Exception as e:
        print(f"⚠️ Training failed: {e}")
        print("Continuing with untrained model...")
    
    # Step 3: Launch fixed interface
    print("\n🚀 Step 3: Launching fixed interface...")
    try:
        from fixed_interface import create_fixed_app
        app = create_fixed_app()
        app.launch(share=True, server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"❌ Interface failed: {e}")

if __name__ == "__main__":
    run_fixed_system()