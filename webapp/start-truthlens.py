#!/usr/bin/env python3
"""
Startup script for TruthLens Model Web Application
This script starts both the frontend (Next.js) and TruthLens model backend (Flask) servers
"""
import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command and return the process."""
    print(f"Running: {command}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    process = subprocess.Popen(
        command,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    return process

def monitor_process(process, name):
    """Monitor a process and print its output."""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[{name}] {line.rstrip()}")
    except Exception as e:
        print(f"Error monitoring {name}: {e}")

def main():
    print("🚀 Starting TruthLens Model Web Application...")
    current_dir = Path(__file__).parent.absolute()
    
    # Check if we're in the right directory
    if not (current_dir / "package.json").exists():
        print("❌ Error: package.json not found. Please run this script from the webapp directory.")
        return
    
    # Check for Node.js and npm
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("✅ Node.js and npm are available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Node.js or npm not found. Please install Node.js first.")
        return
    
    # Check for Python
    try:
        subprocess.run([sys.executable, "--version"], check=True, capture_output=True)
        print("✅ Python is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Python not found.")
        return
    
    # Install frontend dependencies
    print("\n📦 Installing frontend dependencies...")
    try:
        subprocess.run(["npm", "install"], cwd=current_dir, check=True)
        print("✅ Frontend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing frontend dependencies: {e}")
        return
    
    # Install backend dependencies
    print("\n📦 Installing backend dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend-requirements.txt"], cwd=current_dir, check=True)
        print("✅ Backend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing backend dependencies: {e}")
        return
    
    processes = []
    
    try:
        # Start TruthLens Model Backend Server
        print("\n🔧 Starting TruthLens Model Backend Server...")
        backend_process = run_command(f"{sys.executable} truthlens-backend.py", cwd=current_dir)
        processes.append(("TruthLens Backend", backend_process))
        
        # Start monitoring backend
        backend_monitor = threading.Thread(
            target=monitor_process, 
            args=(backend_process, "TruthLens Backend")
        )
        backend_monitor.daemon = True
        backend_monitor.start()
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(5)
        
        # Test backend health
        try:
            import requests
            response = requests.get("http://localhost:8000/api/v1/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Backend is healthy: {health_data}")
            else:
                print(f"⚠️ Backend health check failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not test backend health: {e}")
        
        # Start Frontend Server
        print("\n🌐 Starting Frontend Server...")
        frontend_process = run_command("npm run dev", cwd=current_dir)
        processes.append(("Frontend", frontend_process))
        
        # Start monitoring frontend
        frontend_monitor = threading.Thread(
            target=monitor_process, 
            args=(frontend_process, "Frontend")
        )
        frontend_monitor.daemon = True
        frontend_monitor.start()
        
        print("\n" + "="*60)
        print("🎉 TruthLens Model Web Application is starting!")
        print("="*60)
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend: http://localhost:8000")
        print("🏥 Health Check: http://localhost:8000/api/v1/health")
        print("="*60)
        print("Press Ctrl+C to stop all servers")
        print("="*60)
        
        # Wait for processes to complete
        for name, process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️ {name} force killed")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        
        print("👋 All servers stopped. Goodbye!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        for name, process in processes:
            try:
                process.terminate()
            except:
                pass

if __name__ == "__main__":
    main()
