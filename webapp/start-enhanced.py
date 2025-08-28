#!/usr/bin/env python3
"""
Startup script for Enhanced TruthLens Web Application
This script starts both the frontend (Next.js) and enhanced backend (Flask) servers
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command and return the process"""
    print(f"Running: {command}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    return process

def monitor_process(process, name):
    """Monitor a process and print its output"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{name}] {line.rstrip()}")
    except Exception as e:
        print(f"Error monitoring {name}: {e}")

def main():
    print("🚀 Starting Enhanced TruthLens Web Application...")
    print("=" * 60)
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Check if we're in the webapp directory
    if not (current_dir / "package.json").exists():
        print("❌ Error: This script must be run from the webapp directory")
        sys.exit(1)
    
    # Check if Node.js is installed
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        print("✅ Node.js is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Node.js is not installed or not in PATH")
        print("Please install Node.js from https://nodejs.org/")
        sys.exit(1)
    
    # Check if npm is installed
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("✅ npm is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: npm is not installed or not in PATH")
        sys.exit(1)
    
    # Check if Python is installed
    try:
        subprocess.run([sys.executable, "--version"], check=True, capture_output=True)
        print("✅ Python is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Python is not installed or not in PATH")
        sys.exit(1)
    
    # Install frontend dependencies if needed
    if not (current_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        install_process = run_command("npm install", cwd=current_dir)
        install_process.wait()
        if install_process.returncode != 0:
            print("❌ Error: Failed to install frontend dependencies")
            sys.exit(1)
        print("✅ Frontend dependencies installed")
    
    # Install backend dependencies if needed
    backend_requirements = current_dir / "backend-requirements.txt"
    if backend_requirements.exists():
        print("📦 Installing backend dependencies...")
        install_backend = run_command(f"{sys.executable} -m pip install -r backend-requirements.txt", cwd=current_dir)
        install_backend.wait()
        if install_backend.returncode != 0:
            print("❌ Error: Failed to install backend dependencies")
            sys.exit(1)
        print("✅ Backend dependencies installed")
    
    processes = []
    
    try:
        # Start enhanced backend server
        print("\n🔧 Starting Enhanced Backend Server...")
        backend_process = run_command(f"{sys.executable} enhanced-backend.py", cwd=current_dir)
        processes.append(("Enhanced Backend", backend_process))
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Start frontend server
        print("\n🌐 Starting Frontend Server...")
        frontend_process = run_command("npm run dev", cwd=current_dir)
        processes.append(("Frontend", frontend_process))
        
        # Monitor processes
        threads = []
        for name, process in processes:
            thread = threading.Thread(target=monitor_process, args=(process, name))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        print("\n" + "=" * 60)
        print("🎉 Enhanced TruthLens Web Application is starting up!")
        print("=" * 60)
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Enhanced Backend API: http://localhost:8000")
        print("🏥 Health Check: http://localhost:8000/api/v1/health")
        print("=" * 60)
        print("✨ Enhanced Features:")
        print("   • Better news article analysis")
        print("   • Entity extraction and claim detection")
        print("   • Government source verification")
        print("   • Enhanced evidence search")
        print("=" * 60)
        print("🧪 Test Examples:")
        print("   • 'PM Modi speaks with French President Macron'")
        print("   • 'COVID-19 vaccines cause autism'")
        print("   • '5G technology causes health problems'")
        print("   • 'Climate change is a hoax'")
        print("=" * 60)
        print("Press Ctrl+C to stop all servers")
        print("=" * 60)
        
        # Wait for processes to complete
        for name, process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        
        # Terminate all processes
        for name, process in processes:
            try:
                process.terminate()
                print(f"✅ Stopped {name} server")
            except Exception as e:
                print(f"❌ Error stopping {name} server: {e}")
        
        print("👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Clean up processes
        for name, process in processes:
            try:
                process.terminate()
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()
