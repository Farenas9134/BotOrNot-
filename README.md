# BotOrNot: CS320 Final Project

**Authors: Francisco Arenas & Shalim Montes Hernandez**

### Project Details
Social media bots are automated software that simulate human behavior to perform a range of tasks like boosting stats and engagements on posts via likes, views, and comments. While
these relatively benign goals may appear harmless, automated bots are far more concerning when used for malicious purposes. Recent bots have been tasked with goals such as pushing extremist propaganda, impersonating real people, or attempting to steal personal information from human users by appealing to carnal desires. This combination of malicious-oriented automated programs and increased deployment of these bots calls for a critical analysis of public online activity on social media platforms. 

For this reason, we propose a model that aims to predict whether an account or tweet originates from a human user or an automated bot. We used both the KNN and Random Forest models to construct our distinguisher.

### Steps for Running Code (ETA runtime: ~1 hour):
1. Download Git LFS to load in twiBot-22 dataset (Steps not a full guide) 
    - macOS (using Homebrew): brew install git-lfs
    - Linux (Debian/Ubuntu-based): sudo apt-get install git-lfs
    - Windows: git lfs install
2. Create a virtual environment with python v. 3.11
    - Windows: python -m venv .venv
    - MacOS / Linux: python3.11 -m venv .venv
3. Install required modules into venv:
    - pip install -r requirements.txt
4. Run main manager file in root directory:
    - python manager.py