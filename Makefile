
deps:
	pip-compile requirements.in

install:
	pip install -r requirements.txt

notebook:
	jupyter notebook --no-browser
