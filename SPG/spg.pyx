from libc.stdlib cimport malloc,free
from libc.math cimport round,fabs,sqrt

cimport numpy
cimport cspglib

import numpy

# NOTE: all indices in this module are zero-based.

cdef inline double _c_max(double a,double b):
    "Return the maximum of two numbers."
    if a>b:
        return a
    return b


cdef void _apply_symmetry(double[:,:] rotation,double[:] translation,
                          double[:] pos,double [:] result):
    """
    Compute the result of applying a symmetry operation to a position
    vector. No shape check is performed.
    """
    cdef int i,j
    for i in range(3):
        result[i]=translation[i]
        for j in range(3):
            result[i]+=rotation[i,j]*pos[j]


cdef class SymmetryOperations:
  """
  Object that contains all the interesting information about the
  crystal symmetry group of a set of atoms.
  """
  cdef public double[:,:] __lattvectors
  cdef public int[:] __types
  cdef public double[:,:] __positions
  cdef readonly str symbol
  cdef readonly double[:] __shift
  cdef readonly double[:,:] __transform
  cdef readonly double[:,:,:] __rotations
  cdef readonly double[:,:] __translations
  cdef readonly double[:] __norms
  cdef double c_latvectors[3][3]
  cdef int *c_types
  cdef double (*c_positions)[3]
  cdef int natoms,nsyms
  cdef double symprec

  property lattice_vectors:
      def __get__(self):
          return numpy.asarray(self.__lattvectors)
  property types:
      def __get__(self):
          return numpy.asarray(self.__lattvectors)
  property positions:
      def __get__(self):
          return numpy.asarray(self.__positions)
  property origin_shift:
      def __get__(self):
          return numpy.asarray(self.__shift)
  property transformation_matrix:
      def __get__(self):
          return numpy.asarray(self.__transform)
  property rotations:
      def __get__(self):
          return numpy.asarray(self.__rotations)
  property translations:
      def __get__(self):
          return numpy.asarray(self.__translations)

  cdef void __build_c_arrays(self):
      """
      Build the internal low-level representations of the input
      parameters, ready to be passed to C functions.
      """
      self.c_types=<int*>malloc(self.natoms*sizeof(int))
      self.c_positions=<double(*)[3]>malloc(self.natoms*sizeof(double[3]))
      if self.c_types is NULL or self.c_positions is NULL:
          raise MemoryError()

  cdef void __refresh_c_arrays(self):
      """
      Copy the values of __types, __positions and __lattvectors to
      their C counterparts.
      """
      cdef int i,j
      for i in range(3):
          for j in range(3):
              self.c_latvectors[i][j]=self.__lattvectors[i,j]
      for i in range(self.natoms):
          self.c_types[i]=self.__types[i]
          for j in range(3):
              self.c_positions[i][j]=self.__positions[i,j]

  cdef void __spg_get_dataset(self) except *:
      """
      Thin, slightly selective wrapper around spg_get_dataset(). The
      interesting information is copied out to Python objects and the
      rest discarded.
      """
      cdef int i,j,k
      cdef double tmp
      cdef cspglib.SpglibDataset *data
      data=cspglib.spg_get_dataset(self.c_latvectors,
                                   self.c_positions,
                                   self.c_types,
                                   self.natoms,
                                   self.symprec)
      # The C arrays can get corrupted by this function call.
      self.__refresh_c_arrays()
      if data is NULL:
          raise MemoryError()
      self.symbol=data.international_symbol.encode("ASCII").strip()
      self.__shift=numpy.empty((3,),dtype=numpy.float64)
      self.__transform=numpy.empty((3,3),dtype=numpy.float64)
      self.nsyms=data.n_operations
      self.__rotations=numpy.empty((self.nsyms,3,3),
                                   dtype=numpy.float64)
      self.__translations=numpy.empty((self.nsyms,3),
                                      dtype=numpy.float64)
      for i in range(3):
          self.__shift[i]=data.origin_shift[i]
          for j in range(3):
              self.__transform[i,j]=data.transformation_matrix[i][j]
      for i in range(self.nsyms):
          for j in range(3):
              self.__translations[i,j]=data.translations[i][j]
              for k in range(3):
                  self.__rotations[i,j,k]=data.rotations[i][j][k]
      cspglib.spg_free_dataset(data)

  def __cinit__(self,lattvectors,types,positions,symprec=1e-5):
      cdef int i
      self.__lattvectors=numpy.array(lattvectors,dtype=numpy.float64)
      self.__types=numpy.array(types,dtype=numpy.int32)
      self.__positions=numpy.array(positions,dtype=numpy.float64)
      self.__norms=numpy.empty((3,),dtype=numpy.float64)
      for i in range(3):
          self.__norms[i]=sqrt(lattvectors[i,0]**2+
                               lattvectors[i,1]**2+
                               lattvectors[i,2]**2)
      self.natoms=self.positions.shape[0]
      self.symprec=symprec
      if self.__positions.shape[0]!=self.natoms or self.__positions.shape[1]!=3:
          raise ValueError("positions must be a natoms x 3 array")
      if not (self.__lattvectors.shape[0]==self.__lattvectors.shape[1]==3):
          raise ValueError("lattice vectors must form a 3 x 3 matrix")
      self.__build_c_arrays()
      self.__refresh_c_arrays()
      self.__spg_get_dataset()

  def __dealloc__(self):
      if self.c_types is not NULL:
          free(self.c_types)
      if self.c_positions is not NULL:
          free(self.c_positions)
