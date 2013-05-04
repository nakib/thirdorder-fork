include arch.make

python/thirdorder_core.so: core/cthirdorder_core.pxd core/thirdorder_core.pyx core/thirdorder_fortran.o
	cd core && \
	python setup.py build --build-lib=../python --build-platlib=../python

core/thirdorder_fortran.o: core/thirdorder_fortran.f90
	cd core && \
	$(FC) $(FFLAGS) -o thirdorder_fortran.o -c thirdorder_fortran.f90

clean:
	rm -Rf python/thirdorder_core.so core/*.mod core/*.c core/*.o core/build

