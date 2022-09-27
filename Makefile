c:
	@python3 setup.py build_ext -i

cf:
	@python3 setup.py build_ext -if

r:
	@(cd tests; python3 test.py)

clean:

	@./cleaner
