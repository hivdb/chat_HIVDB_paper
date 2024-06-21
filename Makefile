all:
	./env/bin/python3 work.py

save_env:
	./env/bin/pip3 freeze > requirements.txt

query:
	./env/bin/python3 query_hivdb.py

.PHONY:
	all save_env
