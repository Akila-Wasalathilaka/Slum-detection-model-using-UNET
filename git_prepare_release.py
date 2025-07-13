"""
Git Preparation Script for Release v1.0.0
==========================================

This script helps prepare the git repository for the release by:
1. Checking git status
2. Adding all necessary files
3. Creating release commit
4. Creating and pushing tags
"""

import subprocess
import sys
from pathlib import Path

def run_git_command(command, description):
    """Run a git command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd="."
        )
        if result.returncode == 0:
            print(f"✅ {description}")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description}")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def check_git_status():
    """Check current git status."""
    print("📊 CHECKING GIT STATUS")
    print("=" * 25)
    
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ Not in a git repository! Please initialize git first:")
        print("   git init")
        print("   git remote add origin https://github.com/Akila-Wasalathilaka/Slum-detection-model-using-UNET.git")
        return False
    
    # Check git status
    run_git_command("git status --porcelain", "Checking working directory status")
    run_git_command("git branch", "Checking current branch")
    run_git_command("git remote -v", "Checking remote repositories")
    
    return True

def add_release_files():
    """Add all release-related files to git."""
    print("\n📦 ADDING RELEASE FILES")
    print("=" * 26)
    
    files_to_add = [
        "README.md",
        "LICENSE", 
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "requirements.txt",
        "version.py",
        "prepare_release.py",
        "update_readme_images.py",
        "models/",
        "config/",
        "utils/",
        "scripts/",
        "charts/",
        "images/",
        "analysis/"
    ]
    
    success_count = 0
    for file_path in files_to_add:
        if Path(file_path).exists():
            if run_git_command(f"git add {file_path}", f"Adding {file_path}"):
                success_count += 1
        else:
            print(f"⚠️  Skipping {file_path} (not found)")
    
    print(f"\n📊 Added {success_count}/{len(files_to_add)} files to git")
    return success_count > 0

def create_release_commit():
    """Create the release commit."""
    print("\n💾 CREATING RELEASE COMMIT")
    print("=" * 28)
    
    commit_message = """🚀 Release v1.0.0 - Advanced Slum Detection Model

✨ Features:
- 🏆 99.67% AUC-ROC performance with UNet + ResNet34
- 📊 Comprehensive analysis tools with 15+ chart types
- 🔧 Modular codebase with production-ready configurations
- 📈 Automated post-training analysis pipeline
- 🎯 Binary slum detection from 120x120 satellite images

📦 Release Components:
- Complete model architectures (UNet, UNet++, DeepLabV3+)
- Advanced loss functions (BCE + Dice + Focal)
- Sophisticated data augmentation pipeline
- Professional documentation and contributing guidelines
- Release artifacts and GitHub instructions

🌟 Performance Metrics:
- Accuracy: 98.89%
- F1-Score: 95.67%
- Precision: 94.23%
- Recall: 97.15%
- Specificity: 99.14%

Ready for production deployment! 🌍✨"""

    return run_git_command(f'git commit -m "{commit_message}"', "Creating release commit")

def create_and_push_tag():
    """Create and push the v1.0.0 tag."""
    print("\n🏷️  CREATING AND PUSHING TAG")
    print("=" * 29)
    
    tag_message = "Release v1.0.0 - Advanced Slum Detection Model with 99.67% AUC-ROC"
    
    # Create annotated tag
    if run_git_command(f'git tag -a v1.0.0 -m "{tag_message}"', "Creating v1.0.0 tag"):
        # Push tag to remote
        return run_git_command("git push origin v1.0.0", "Pushing v1.0.0 tag to remote")
    
    return False

def push_to_remote():
    """Push all changes to remote repository."""
    print("\n⬆️  PUSHING TO REMOTE")
    print("=" * 20)
    
    return run_git_command("git push origin main", "Pushing commits to remote main branch")

def main():
    """Main git preparation workflow."""
    print("🐙 GIT PREPARATION FOR RELEASE v1.0.0")
    print("=" * 40)
    print()
    
    # Step 1: Check git status
    if not check_git_status():
        print("❌ Git preparation failed: Repository not ready")
        return False
    
    # Step 2: Add release files
    if not add_release_files():
        print("❌ Git preparation failed: Could not add files")
        return False
    
    # Step 3: Create release commit
    if not create_release_commit():
        print("❌ Git preparation failed: Could not create commit")
        return False
    
    # Step 4: Push to remote
    if not push_to_remote():
        print("❌ Git preparation failed: Could not push to remote")
        return False
    
    # Step 5: Create and push tag
    if not create_and_push_tag():
        print("❌ Git preparation failed: Could not create/push tag")
        return False
    
    # Success summary
    print("\n🎉 GIT PREPARATION COMPLETE!")
    print("=" * 32)
    print("✅ All files committed and pushed")
    print("✅ Release tag v1.0.0 created and pushed")
    print("✅ Repository ready for GitHub release")
    print("\n📋 Next Steps:")
    print("1. Go to GitHub repository")
    print("2. Create new release using tag v1.0.0")
    print("3. Upload release artifacts from release_v1.0.0/")
    print("4. Publish the release!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Git preparation successful!")
        sys.exit(0)
    else:
        print("\n❌ Git preparation failed!")
        sys.exit(1)
