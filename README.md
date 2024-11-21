# SoccerAssistant


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

1. Running the ChatAgent
   ```bash
   export PYTHONPATH=$(pwd):PYTHONPATH
   python chat_agent/app.py
   ```

The app will be running on http://127.0.0.1:7860

2. Running the SoccerStrat ChatAgent
   ```bash
   cd strategy_agent/SoccerStrat
   streamlit run soccer_app.py
   ```

The app will be running on http://localhost:8501


