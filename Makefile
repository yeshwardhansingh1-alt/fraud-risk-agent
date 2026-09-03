.PHONY: data features train evaluate serve dashboard all test

data:
	python notebooks/01_load_and_inspect.py
	python notebooks/02_eda.py

features:
	python features/build_features.py

train:
	python model/train.py
	python model/calibrate.py

evaluate:
	python model/rule_engine.py
	python model/evaluate.py
	python model/net_financial_impact.py

serve:
	python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

dashboard:
	streamlit run dashboard.py

test:
	pytest tests/

all: data features train evaluate
