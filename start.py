#!/usr/bin/env python3
"""
LeafScan - Crop Disease Detection
Quick startup script
"""
import subprocess, sys, os

print("🌿 LeafScan - Crop Disease Detection System")
print("=" * 50)

print("\n🚀 Starting Flask server on http://localhost:5001")
print("   Open your browser to: http://localhost:5001")
print("   Press Ctrl+C to stop\n")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "backend/app.py"])
