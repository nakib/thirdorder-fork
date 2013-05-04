include arch.make

PYTHON/spg.so: SPG/cspglib.pxd SPG/spg.pyx
	cd SPG && \
	python setup.py build --build-lib=../PYTHON --build-platlib=../PYTHON

clean:
	rm -f PYTHON/spg.so

