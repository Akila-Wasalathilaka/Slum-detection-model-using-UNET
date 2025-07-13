"""
Repository Cleanup and Organization Script for v1.0.0 Release
============================================================

This script cleans up the repository by:
1. Removing unwanted/temporary files
2. Moving files to correct locations
3. Organizing the final repository structure
4. Keeping only production-ready files

Based on chat history analysis, this ensures a clean v1.0.0 release.
"""

import os
import shutil
import glob
from pathlib import Path

def remove_unwanted_files():
    """Remove temporary, test, and unwanted files."""
    print("🧹 REMOVING UNWANTED FILES")
    print("=" * 28)
    
    # Files to remove (temporary, test, legacy)
    files_to_remove = [
        # Temporary and test files
        "*.pyc",
        "*.pyo", 
        "*.pyd",
        "__pycache__/",
        ".pytest_cache/",
        "*.tmp",
        "*.temp",
        
        # Development files (keep only if needed)
        "prepare_release.py",  # Can remove after release
        "git_prepare_release.py",  # Can remove after release
        "release_summary.py",  # Can remove after release
        "update_readme_images.py",  # Keep this one - useful for updates
        
        # Legacy analysis files (already moved to analysis/)
        "test_analysis.py",
        "debug_*.py",
        "temp_*.py",
        
        # OS specific files
        "Thumbs.db",
        ".DS_Store",
        "desktop.ini",
        
        # Editor files
        "*.swp",
        "*.swo",
        "*~",
        ".vscode/settings.json"  # Keep .vscode but remove local settings
    ]
    
    removed_count = 0
    for pattern in files_to_remove:
        if "*" in pattern:
            # Handle glob patterns
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"✅ Removed file: {file_path}")
                        removed_count += 1
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"✅ Removed directory: {file_path}")
                        removed_count += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")
        else:
            # Handle specific files
            if os.path.exists(pattern):
                try:
                    if os.path.isfile(pattern):
                        os.remove(pattern)
                        print(f"✅ Removed file: {pattern}")
                        removed_count += 1
                    elif os.path.isdir(pattern):
                        shutil.rmtree(pattern)
                        print(f"✅ Removed directory: {pattern}")
                        removed_count += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {pattern}: {e}")
    
    print(f"\n📊 Removed {removed_count} unwanted files/directories")
    return removed_count

def organize_repository_structure():
    """Ensure proper repository structure."""
    print("\n📁 ORGANIZING REPOSITORY STRUCTURE")
    print("=" * 36)
    
    # Required directories
    required_dirs = [
        "models",
        "config", 
        "utils",
        "scripts",
        "charts",
        "experiments",
        "analysis",
        "images",
        "data"
    ]
    
    # Create missing directories
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    return True

def move_misplaced_files():
    """Move files to their correct locations."""
    print("\n🔄 MOVING MISPLACED FILES")
    print("=" * 26)
    
    # Files that should be moved
    moves = {
        # Any remaining analysis files to analysis/
        "analyze_*.py": "analysis/",
        "mask_analysis*.py": "analysis/", 
        "data_exploration*.py": "analysis/",
        
        # Documentation files (keep in root)
        # These are already in the right place
        
        # Model files (should be in models/)
        # These are already properly organized
        
        # Config files (should be in config/)
        # These are already properly organized
    }
    
    moved_count = 0
    for pattern, destination in moves.items():
        if "*" in pattern:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    dest_path = os.path.join(destination, os.path.basename(file_path))
                    try:
                        # Create destination directory if it doesn't exist
                        os.makedirs(destination, exist_ok=True)
                        
                        # Move file if it's not already there
                        if not os.path.exists(dest_path):
                            shutil.move(file_path, dest_path)
                            print(f"✅ Moved: {file_path} → {dest_path}")
                            moved_count += 1
                        else:
                            print(f"⚠️  Destination exists: {dest_path}")
                    except Exception as e:
                        print(f"❌ Could not move {file_path}: {e}")
    
    print(f"\n📊 Moved {moved_count} files to correct locations")
    return moved_count

def clean_empty_directories():
    """Remove empty directories."""
    print("\n🗂️  CLEANING EMPTY DIRECTORIES")
    print("=" * 30)
    
    removed_count = 0
    for root, dirs, files in os.walk(".", topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # Skip important directories even if empty
                skip_dirs = {".git", ".vscode", "data", "experiments", "release_v1.0.0"}
                if dir_name in skip_dirs:
                    continue
                
                # Check if directory is empty
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"✅ Removed empty directory: {dir_path}")
                    removed_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {dir_path}: {e}")
    
    print(f"\n📊 Removed {removed_count} empty directories")
    return removed_count

def validate_core_files():
    """Validate that all core files are present."""
    print("\n✅ VALIDATING CORE FILES")
    print("=" * 25)
    
    # Essential files that must exist
    essential_files = [
        "README.md",
        "LICENSE",
        "CHANGELOG.md", 
        "CONTRIBUTING.md",
        "SECURITY.md",
        "requirements.txt",
        "version.py",
        
        # Core Python modules
        "models/__init__.py",
        "models/unet.py",
        "models/losses.py", 
        "models/metrics.py",
        
        "config/__init__.py",
        "config/model_config.py",
        "config/training_config.py",
        "config/data_config.py",
        
        "utils/__init__.py",
        "utils/dataset.py",
        "utils/transforms.py",
        "utils/visualization.py",
        "utils/checkpoint.py",
        
        "scripts/train.py",
        "scripts/test.py", 
        "scripts/inference.py",
        
        "charts/post_training_analysis.py",
        "charts/model_analysis.py",
        "charts/quick_analysis.py"
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in essential_files:
        if os.path.exists(file_path):
            present_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    print(f"\n📊 Validation Results:")
    print(f"✅ Present: {len(present_files)}/{len(essential_files)} files")
    if missing_files:
        print(f"❌ Missing: {len(missing_files)} files")
        print("Missing files:", missing_files)
        return False
    
    return True

def create_final_summary():
    """Create a summary of the cleaned repository."""
    print("\n📋 FINAL REPOSITORY STRUCTURE")
    print("=" * 32)
    
    # Count files in each directory
    structure = {}
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and release folder
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'release_v1.0.0']
        
        if root == ".":
            structure["Root"] = len([f for f in files if not f.startswith('.')])
        else:
            folder_name = root.replace("./", "").replace("\\", "/")
            if "/" not in folder_name:  # Only top-level folders
                structure[folder_name] = len(files)
    
    for folder, count in sorted(structure.items()):
        print(f"📁 {folder}: {count} files")
    
    # Calculate total
    total_files = sum(structure.values())
    print(f"\n📊 Total files: {total_files}")
    
    return structure

def main():
    """Main cleanup workflow."""
    print("🧹 REPOSITORY CLEANUP FOR v1.0.0 RELEASE")
    print("=" * 45)
    print("🎯 Goal: Clean, organized, production-ready repository")
    print()
    
    try:
        # Step 1: Remove unwanted files
        removed_files = remove_unwanted_files()
        
        # Step 2: Organize directory structure
        organize_repository_structure()
        
        # Step 3: Move misplaced files
        moved_files = move_misplaced_files()
        
        # Step 4: Clean empty directories
        removed_dirs = clean_empty_directories()
        
        # Step 5: Validate core files
        validation_passed = validate_core_files()
        
        # Step 6: Create final summary
        structure = create_final_summary()
        
        # Final report
        print("\n🎉 CLEANUP COMPLETE!")
        print("=" * 20)
        print(f"✅ Files removed: {removed_files}")
        print(f"✅ Files moved: {moved_files}")
        print(f"✅ Empty directories removed: {removed_dirs}")
        print(f"✅ Core files validated: {'Yes' if validation_passed else 'No'}")
        
        if validation_passed:
            print("\n🚀 Repository is ready for v1.0.0 release!")
            print("📦 All essential files are present and organized")
            print("🎯 Ready for production deployment")
        else:
            print("\n⚠️  Some core files are missing!")
            print("Please check the validation results above")
        
        return validation_passed
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Repository cleanup successful!")
    else:
        print("\n❌ Repository cleanup had issues!")
        print("Please review the output above and fix any missing files.")
