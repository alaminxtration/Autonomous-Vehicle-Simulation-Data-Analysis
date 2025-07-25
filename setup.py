#!/usr/bin/env python3
"""
Autonomous Vehicle Simulation Data Analysis Package
A comprehensive simulation and analysis framework for autonomous vehicle sensor data
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="autonomous-vehicle-simulation",
    version="1.0.0",
    author="alaminxtration",
    author_email="your-email@example.com",  # Update with your email
    description="A comprehensive simulation and analysis framework for autonomous vehicle sensor data",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis",
    project_urls={
        "Bug Tracker": "https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis/issues",
        "Documentation": "https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis#readme",
        "Source Code": "https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docker": [
            "docker-compose>=1.25",
        ],
        "full": [
            "jupyter>=1.0",
            "matplotlib>=3.3",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "av-simulate=simulation.vehicle_simulation:main",
            "av-dashboard=dashboard.simple_dashboard:main",
            "av-status=deployment_status:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "autonomous-vehicles",
        "simulation",
        "data-analysis",
        "machine-learning",
        "sensor-data",
        "real-time",
        "dashboard",
        "docker",
        "redis",
    ],
)
