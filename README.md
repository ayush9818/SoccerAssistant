# SoccerAssistant

## Overview

SoccerAssistant is a two-system chatbot we created for the Northwestern men's soccer team. It is designed to enhance performance both on and off the field. The first system focuses on **competitive strategy**, helping coaches and analysts review opponent tactics, assess strengths and weaknesses, and recommend strategies for upcoming matches. The second system is dedicated to **analytics**, providing detailed insights into player and team performance, aiding in **player evaluation** by tracking key metrics, and helping with **team evaluation** to identify areas for improvement. Together, these systems support the team’s decision-making process, helping the coaching staff make data-driven adjustments to improve both individual and team performance.

## Key Features

- ChatBot UI with 4-agent backend
- Quickly generate summary statistics, player comparisons, and data visualizations
- Identifies potential weaknesses in opposing teams for scouting and gameplan purposes
- Tracks and analyzes player performance metrics (e.g., goals, assists, passing accuracy).
- Provides data on team performance, including possession, shots on target, and defensive effectiveness.
- Identifies areas for improvement based on collective team data. 

## Setup Instructions

Follow these steps to set up and run the SoccerAssistant project:

### 1. Clone the Repository
```bash
git clone https://github.com/ayush9818/SoccerAssistant.git
cd SoccerAssistant
```


### 2. Set Up a Virtual Environment
**For Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root directory and add the following variables:

```bash
cp env_template .env
```

- OPENAI_API_KEY : Replace it with appropriate API Key

### 4. How to Run

1. Run the application:
   ```bash
   export PYTHONPATH=$(pwd):PYTHONPATH
   python chat_agent/app.py
   ```

The app will be running on http://127.0.0.1:7860.
