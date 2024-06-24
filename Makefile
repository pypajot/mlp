install:
	pip install -r requirements.txt

mclean:
	rm -f metrics.pkl

fclean:
	rm -f *.pkl
	rm -rf data_train.csv data_validation.csv
.PHONY: install