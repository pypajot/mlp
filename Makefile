install:
	pip install -r requirements.txt

mclean:
	rm -f metrics.pkl

fclean:
	rm -f *.pkl
	rm -f data_train.csv
	rm -f data_validation.csv
.PHONY: install