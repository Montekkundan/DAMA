# Setup

## setup venv enviroment 

```bash
python -m venv venv
```

for mac: 
```bash
source venv/bin/activate
```

for windows: 
```bash
./venv/Scripts/activate
```

## install requirements

```bash
pip install -r requirements.txt
```

# How to run

```bash
python chat/main.py
```

# Testing

in root directory run:

```bash
pytest
```

for debugging:

```bash
pytest -s
```