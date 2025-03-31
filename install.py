import os
import subprocess
import sys
import venv

def setup_virtual_environment():
    """Create and setup virtual environment"""
    venv.create('venv', with_pip=True)
    
    # Determine the pip path
    if sys.platform == 'win32':
        pip_path = os.path.join('venv', 'Scripts', 'pip')
    else:
        pip_path = os.path.join('venv', 'bin', 'pip')
    
    # Install requirements
    subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])

def create_startup_script():
    """Create startup scripts for different platforms"""
    if sys.platform == 'win32':
        # Windows batch file
        with open('start_app.bat', 'w') as f:
            f.write('@echo off\n')
            f.write('call venv\\Scripts\\activate.bat\n')
            f.write('python dashboard.py\n')
            f.write('pause\n')
    else:
        # Unix shell script
        with open('start_app.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('source venv/bin/activate\n')
            f.write('python dashboard.py\n')
        os.chmod('start_app.sh', 0o755)

def main():
    print("Setting up Jewelry Shop Security System...")
    
    # Create virtual environment and install dependencies
    print("Creating virtual environment...")
    setup_virtual_environment()
    
    # Create startup script
    print("Creating startup script...")
    create_startup_script()
    
    print("\nInstallation complete!")
    print("To start the application:")
    if sys.platform == 'win32':
        print("Double-click 'start_app.bat'")
    else:
        print("Run './start_app.sh'")

if __name__ == '__main__':
    main() 