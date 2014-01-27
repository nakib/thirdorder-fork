!  thirdorder, help compute anharmonic IFCs from minimal sets of displacements
!  Copyright (C) 2012-2013 Wu Li <wu.li.phys2011@gmail.com>
!  Copyright (C) 2012-2013 Jesús Carrete Montaña <jcarrete@gmail.com>
!  Copyright (C) 2012-2013 Natalio Mingo Bisquert <natalio.mingo@cea.fr>
!
!  This program is free software: you can redistribute it and/or modify
!  it under the terms of the GNU General Public License as published by
!  the Free Software Foundation, either version 3 of the License, or
!  (at your option) any later version.
!
!  This program is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!  GNU General Public License for more details.
!
!  You should have received a copy of the GNU General Public License
!  along with this program.  If not, see <http://www.gnu.org/licenses/>.

! The computationally intensive parts of the algorithm are implemented here.
! wedge() is the core of this file, but the subroutine gaussian() is
! also used from outside.
! Note that explicitly C-compatible types are used. This is critical to avoid
! crashes with some {architecture/C compiler/Fortran compiler} combinations.

module thirdorder_fortran
  use iso_c_binding
  implicit none

contains

  ! Determine a minimal set of third-order derivatives of the energy
  ! needed to obtain all anharmonic IFCs withing the cutoff radius
  ! ForceRange. The description of the constants is returned in cList;
  ! the rest of the output arguments are necessary for the
  ! reconstruction since they describe the equivalences and
  ! transformation rules between atomic triplets.
  subroutine wedge(LatVec,Coord,CoordAll,Orth,Trans,Natoms,Nlist,cNequi,cList,&
       cALLEquiList,cTransformationArray,cNIndependentBasis,&
       cIndependentBasis,Ngrid1,Ngrid2,Ngrid3,Nsymm,&
       ForceRange,Allocsize) bind(C,name="wedge")
    implicit none

    real(kind=C_DOUBLE),value,intent(in) :: ForceRange
    real(kind=C_DOUBLE),intent(in) :: LatVec(3,3),Orth(3,3,Nsymm),Trans(3,Nsymm)
    integer(kind=C_INT),value,intent(in) :: Natoms,Ngrid1,Ngrid2,Ngrid3,Nsymm
    real(kind=C_DOUBLE),intent(in) :: Coord(3,Natoms),CoordAll(3,Natoms*Ngrid1*Ngrid2*Ngrid3)
    integer(kind=C_INT),intent(out) :: NList,Allocsize
    type(C_PTR),intent(out) :: cNequi,cList,cALLEquiList,&
         cTransformationArray,cNindependentBasis,cIndependentBasis

    integer(kind=C_INT) :: AllAllocsize,NAllList,Nnonzero,EquiList(3,Nsymm*6)
    integer(kind=C_INT),allocatable,target,save :: Nequi(:),ALLEquiList(:,:,:),&
         NIndependentBasis(:),IndependentBasis(:,:),List(:,:)
    integer(kind=C_INT),allocatable :: AllList(:,:)
    real(kind=C_DOUBLE),allocatable,target,save :: TransformationArray(:,:,:,:)
    real(kind=C_DOUBLE),allocatable :: Transformation(:,:,:,:),&
         TransformationAux(:,:,:)
    real(kind=C_DOUBLE),allocatable :: Coeffi_reduced(:,:)
    integer(kind=C_INT),allocatable :: Nequi2(:),ALLEquiList2(:,:,:),&
         NIndependentBasis2(:),IndependentBasis2(:,:),List2(:,:),&
         AllList2(:,:)
    real(kind=C_DOUBLE),allocatable :: TransformationArray2(:,:,:,:),Transformation2(:,:,:,:),&
         TransformationAux2(:,:,:)
    real(kind=C_DOUBLE) ::  Coeffi(6*Nsymm*27,27),BB(27,27)

    integer(kind=C_INT) :: ii,jj,kk,ll,mm,iaux,jaux,Ntot,ipermutation,isym
    integer(kind=C_INT) :: ix,iy,iz,summ
    integer(kind=C_INT) :: ibasis,jbasis,kbasis,ibasisprime,jbasisprime,kbasisprime
    integer(kind=C_INT) :: indexijk,indexijkprime,indexrow,ibasispermut,jbasispermut,kbasispermut
    integer(kind=C_INT) :: triplet(3),triplet_permutation(3),triplet_sym(3)
    integer(kind=C_INT) :: vec1(3),ispecies1,vec2(3),ispecies2,vec3(3),ispecies3
    integer(kind=C_INT) :: ID_equi(Nsymm,Natoms*Ngrid1*Ngrid2*Ngrid3)
    integer(kind=C_INT) :: Ind_cell(3,Natoms*Ngrid1*Ngrid2*Ngrid3),Ind_species(Natoms*Ngrid1*Ngrid2*Ngrid3)
    integer(kind=C_INT) :: shift2(3),shift3(3),shift2all(3,27),shift3all(3,27),N2equi,N3equi
    real(kind=C_DOUBLE) :: dist,dist1,dist_min,Car2(3),Car3(3)
    logical :: NonZero

    real(kind=C_DOUBLE) :: dnrm2

    Allocsize=64
    AllAllocsize=256

    allocate(Nequi(Allocsize))
    allocate(AllEquiList(3,Nsymm*6,Allocsize))
    allocate(TransformationArray(27,27,Nsymm*6,Allocsize))
    allocate(Transformation(27,27,Nsymm*6,Allocsize))
    allocate(TransformationAux(27,27,AllocSize))
    allocate(NIndependentBasis(AllocSize))
    allocate(IndependentBasis(27,Allocsize))
    allocate(List(3,Allocsize))
    allocate(AllList(3,AllAllocsize))

    Ntot=0
    NList=0
    NAllList=0
    List=0
    AllList=0
    summ=0
    TransformationArray=0.d0

    ! Symmetry operations are applied to the set of atomic positions
    ! to determine which is mapped into which by each operation.
    call symmetry(Nsymm,Natoms,LatVec,Coord,ID_equi,&
         Ngrid1,Ngrid2,Ngrid3,Orth,Trans)
    call Id2Ind(Ind_cell,Ind_species,Ngrid1,Ngrid2,Ngrid3,Natoms)

    shift2all=0
    shift3all=0
    ! Each atom in the unit cell (ii) and all atoms in the supercell
    ! (jj,kk) are considered, and the minimal distances between them
    ! computed by taking into account the periodic boundary
    ! conditions. Interactions outside ForceRange are ignored.
    do ii=1,Natoms
       do jj=1,Ngrid1*Ngrid2*Ngrid3*Natoms
          dist_min=huge(dist)
          N2equi=0
          do ix=-1,1
             do iy=-1,1
                do iz=-1,1
                   dist=dnrm2(3,ix*Ngrid1*LatVec(1:3,1)+&
                        iy*Ngrid2*LatVec(1:3,2)+&
                        iz*Ngrid3*LatVec(1:3,3)+&
                        CoordAll(1:3,jj)-CoordAll(1:3,ii),1)
                   if(dist.lt.dist_min) then
                      dist_min=dist
                   end if
                end do
             end do
          end do
          do ix=-1,1
             do iy=-1,1
                do iz=-1,1
                   dist=dnrm2(3,ix*Ngrid1*LatVec(1:3,1)+&
                        iy*Ngrid2*LatVec(1:3,2)+&
                        iz*Ngrid3*LatVec(1:3,3)+&
                        CoordAll(1:3,jj)-CoordAll(1:3,ii),1)
                   if(abs(dist-dist_min).lt.1.d-2) then
                      N2equi=N2equi+1
                      shift2all(:,N2equi)=(/ix,iy,iz/)
                   end if
                end do
             end do
          end do
          dist=dist_min
          if (dist.lt.ForceRange) then
             do kk=1,Ngrid1*Ngrid2*Ngrid3*Natoms
                dist_min=huge(dist)
                N3equi=0
                do ix=-1,1
                   do iy=-1,1
                      do iz=-1,1
                         dist=dnrm2(3,ix*Ngrid1*LatVec(1:3,1)+&
                              iy*Ngrid2*LatVec(1:3,2)+&
                              iz*Ngrid3*LatVec(1:3,3)+&
                              CoordAll(1:3,kk)-CoordAll(1:3,ii),1)
                         if(dist.lt.dist_min) then
                            dist_min=dist
                         end if
                      end do
                   end do
                end do
                do ix=-1,1
                   do iy=-1,1
                      do iz=-1,1
                         dist=dnrm2(3,ix*Ngrid1*LatVec(1:3,1)+&
                              iy*Ngrid2*LatVec(1:3,2)+&
                              iz*Ngrid3*LatVec(1:3,3)+&
                              CoordAll(1:3,kk)-CoordAll(1:3,ii),1)
                         if(abs(dist-dist_min).lt.1.d-2) then
                            N3equi=N3equi+1
                            shift3all(:,N3equi)=(/ix,iy,iz/)
                         end if
                      end do
                   end do
                end do
                dist=dist_min
                dist_min=huge(dist)
                do iaux=1,N2equi
                   Car2=shift2all(1,iaux)*Ngrid1*LatVec(:,1)+&
                        shift2all(2,iaux)*Ngrid2*LatVec(:,2)+&
                        shift2all(3,iaux)*Ngrid3*LatVec(:,3)+CoordAll(1:3,jj)
                   do jaux=1,N3equi
                      Car3=shift3all(1,jaux)*Ngrid1*LatVec(:,1)+&
                           shift3all(2,jaux)*Ngrid2*LatVec(:,2)+&
                           shift3all(3,jaux)*Ngrid3*LatVec(:,3)+CoordAll(1:3,kk)
                      dist1=dnrm2(3,Car3-Car2,1)
                      if(dist1.lt.dist_min) then
                         dist_min=dist1
                         shift2=shift2all(:,iaux)
                         shift3=shift3all(:,jaux)
                      end if
                   end do
                end do
                dist1=dist_min
                if(dist.lt.ForceRange.and.dist1.lt.ForceRange) then
                   ! Atomic triplets are grouped into equivalence
                   ! classes.  Then, the equality of mixed partials
                   ! and point-group symmetries are used to derive
                   ! linear constraints over the anharmonic IFCs of
                   ! one triplet from each class. Finally, Gaussian
                   ! elimination is used, both to extract an
                   ! irreducible subset of constants and to compute
                   ! the matrices linking them to the remaining ones.
                   summ=summ+1
                   iaux=1
                   triplet=(/ii,jj,kk/)
                   if (.not.all(triplet.eq.1)) then
                      do mm=1,NAllList
                         if (all(triplet.eq.AllList(:,mm))) then
                            iaux=0
                         end if
                      end do
                   end if
                   if (iaux.eq.1) then
                      Nlist=Nlist+1
                      if (Nlist.gt.Allocsize) then
                         allocate(Nequi2(2*Allocsize))
                         allocate(AllEquiList2(3,Nsymm*6,2*Allocsize))
                         allocate(TransformationArray2(27,27,Nsymm*6,2*Allocsize))
                         allocate(Transformation2(27,27,Nsymm*6,2*Allocsize))
                         allocate(TransformationAux2(27,27,2*AllocSize))
                         allocate(NIndependentBasis2(2*AllocSize))
                         allocate(IndependentBasis2(27,2*Allocsize))
                         allocate(List2(3,2*Allocsize))
                         Nequi2(1:Allocsize)=Nequi(1:Allocsize)
                         AllEquiList2(:,:,1:AllocSize)=AllEquiList(:,:,1:AllocSize)
                         TransformationArray2(:,:,:,1:Allocsize)=TransformationArray(:,:,:,1:Allocsize)
                         Transformation2(:,:,:,1:Allocsize)=Transformation(:,:,:,1:Allocsize)
                         TransformationAux2(:,:,1:AllocSize)=TransformationAux(:,:,1:AllocSize)
                         NIndependentBasis2(1:AllocSize)=NIndependentBasis(1:AllocSize)
                         IndependentBasis2(:,1:Allocsize)=IndependentBasis(:,1:Allocsize)
                         List2(:,1:Allocsize)=List(:,1:Allocsize)
                         call move_alloc(Nequi2,Nequi)
                         call move_alloc(AllEquiList2,AllEquiList)
                         call move_alloc(TransformationArray2,TransformationArray)
                         call move_alloc(Transformation2,Transformation)
                         call move_alloc(TransformationAux2,TransformationAux)
                         call move_alloc(NIndependentBasis2,NIndependentBasis)
                         call move_alloc(IndependentBasis2,IndependentBasis)
                         call move_alloc(List2,List)
                         Allocsize=Allocsize*2
                      end if
                      List(:,Nlist)=(/ii,jj,kk/)
                      Nequi(Nlist)=0
                      Coeffi=0.d0
                      Nnonzero=0
                      do ipermutation=1,6
                         select case(ipermutation)
                         case(1)
                            triplet_permutation=triplet
                         case(2)
                            triplet_permutation(1)=triplet(2)
                            triplet_permutation(2)=triplet(1)
                            triplet_permutation(3)=triplet(3)
                         case(3)
                            triplet_permutation(1)=triplet(3)
                            triplet_permutation(2)=triplet(2)
                            triplet_permutation(3)=triplet(1)
                         case(4)
                            triplet_permutation(1)=triplet(1)
                            triplet_permutation(2)=triplet(3)
                            triplet_permutation(3)=triplet(2)
                         case(5)
                            triplet_permutation(1)=triplet(2)
                            triplet_permutation(2)=triplet(3)
                            triplet_permutation(3)=triplet(1)
                         case(6)
                            triplet_permutation(1)=triplet(3)
                            triplet_permutation(2)=triplet(1)
                            triplet_permutation(3)=triplet(2)
                         end select
                         do isym=1,Nsymm
                            triplet_sym=(/ID_equi(isym,triplet_permutation(1)),&
                                 ID_equi(isym,triplet_permutation(2)),&
                                 ID_equi(isym,triplet_permutation(3))/)

                            vec1=Ind_cell(:,ID_equi(isym,triplet_permutation(1)))
                            ispecies1=Ind_species(ID_equi(isym,triplet_permutation(1)))
                            vec2=Ind_cell(:,ID_equi(isym,triplet_permutation(2)))
                            ispecies2=Ind_species(ID_equi(isym,triplet_permutation(2)))
                            vec3=Ind_cell(:,ID_equi(isym,triplet_permutation(3)))
                            ispecies3=Ind_species(ID_equi(isym,triplet_permutation(3)))
                            if(.not.all(vec1.eq.0)) then
                               triplet_sym(1)=Ind2Id((/modulo(vec1(1)-vec1(1),&
                                    Ngrid1),modulo(vec1(2)-vec1(2),Ngrid2),&
                                    modulo(vec1(3)-vec1(3),Ngrid3)/),&
                                    ispecies1,Ngrid1,Ngrid2,Natoms)
                               triplet_sym(2)=Ind2Id((/modulo(vec2(1)-vec1(1),&
                                    Ngrid1),modulo(vec2(2)-vec1(2),Ngrid2),&
                                    modulo(vec2(3)-vec1(3),Ngrid3)/),&
                                    ispecies2,Ngrid1,Ngrid2,Natoms)
                               triplet_sym(3)=Ind2Id((/modulo(vec3(1)-vec1(1),&
                                    Ngrid1),modulo(vec3(2)-vec1(2),Ngrid2),&
                                    modulo(vec3(3)-vec1(3),Ngrid3)/),&
                                    ispecies3,Ngrid1,Ngrid2,Natoms)
                            end if
                            do ibasisprime=1,3
                               do jbasisprime=1,3
                                  do kbasisprime=1,3
                                     indexijkprime=(ibasisprime-1)*9+(jbasisprime-1)*3+kbasisprime
                                     indexrow=(ipermutation-1)*Nsymm*27+(isym-1)*27+indexijkprime
                                     do ibasis=1,3
                                        do jbasis=1,3
                                           do kbasis=1,3
                                              indexijk=(ibasis-1)*9+(jbasis-1)*3+kbasis
                                              select case(ipermutation)
                                              case(1)
                                                 ibasispermut=ibasis
                                                 jbasispermut=jbasis
                                                 kbasispermut=kbasis
                                              case(2)
                                                 ibasispermut=jbasis
                                                 jbasispermut=ibasis
                                                 kbasispermut=kbasis
                                              case(3)
                                                 ibasispermut=kbasis
                                                 jbasispermut=jbasis
                                                 kbasispermut=ibasis
                                              case(4)
                                                 ibasispermut=ibasis
                                                 jbasispermut=kbasis
                                                 kbasispermut=jbasis
                                              case(5)
                                                 ibasispermut=jbasis
                                                 jbasispermut=kbasis
                                                 kbasispermut=ibasis
                                              case(6)
                                                 ibasispermut=kbasis
                                                 jbasispermut=ibasis
                                                 kbasispermut=jbasis
                                              end select
                                              BB(indexijkprime,indexijk)=Orth(ibasisprime,ibasispermut,isym)*&
                                                   Orth(jbasisprime,jbasispermut,isym)*&
                                                   Orth(kbasisprime,kbasispermut,isym)
                                           end do
                                        end do
                                     end do
                                  end do
                               end do
                            end do
                            iaux=1
                            if(.not.((ipermutation.eq.1).and.(isym.eq.1))) then
                               do ll=1,Nequi(Nlist)
                                  if(all(triplet_sym.eq.EquiList(:,ll))) then
                                     iaux=0
                                  end if
                               end do
                            end if
                            if (iaux.eq.1) then
                               if(((ipermutation.eq.1).and.(isym.eq.1))&
                                    .or.(.not.all(triplet_sym.eq.triplet))) then
                                  Nequi(Nlist)=Nequi(Nlist)+1
                                  EquiList(:,Nequi(Nlist))=triplet_sym
                                  ALLEquiList(:,Nequi(Nlist),Nlist)=triplet_sym
                                  NAllList=NAllList+1
                                  if(NAllList.ge.AllAllocsize) then
                                     allocate(AllList2(3,2*AllAllocsize))
                                     AllList2(:,1:AllAllocsize)=AllList(:,1:AllAllocsize)
                                     call move_alloc(AllList2,AllList)
                                     AllAllocsize=2*AllAllocsize
                                  end if
                                  AllList(:,NAllList)=triplet_sym
                                  Transformation(:,:,Nequi(Nlist),NList)=BB
                               end if
                            end if
                            if(all(triplet_sym.eq.triplet)) then
                               do indexijkprime=1,27
                                  NonZero=.false.
                                  do indexijk=1,27
                                     if (indexijkprime.eq.indexijk) then
                                        BB(indexijkprime,indexijk)=BB(indexijkprime,indexijk)-1.d0
                                     end if
                                     if (abs(BB(indexijkprime,indexijk)).gt.1.d-12) then
                                        NonZero=.true.
                                     else
                                        BB(indexijkprime,indexijk)=0.d0
                                     end if
                                  end do
                                  if(Nonzero) then
                                     Nnonzero=Nnonzero+1
                                     Coeffi(Nnonzero,1:27)=BB(indexijkprime,1:27)
                                  end if
                               end do
                            end if
                         end do
                      end do
                      allocate(Coeffi_reduced(max(Nnonzero,27),27))
                      Coeffi_reduced=0.d0
                      Coeffi_reduced(1:Nnonzero,1:27)=Coeffi(1:Nnonzero,1:27)
                      call gaussian(Coeffi_reduced,max(Nnonzero,27),27,jaux,&
                           TransformationAux(1:27,1:27,NList),&
                           NIndependentBasis(NList),IndependentBasis(1:27,NList))
                      deallocate(Coeffi_reduced)
                   end if
                end if
             end do
          end if
       end do
    end do
    do ii=1,Nlist
       do jj=1,Nequi(ii)
          TransformationArray(:,1:NIndependentBasis(ii),jj,ii)=&
               matmul(Transformation(:,:,jj,ii),&
               TransformationAux(:,1:NIndependentBasis(ii),ii))
          do kk=1,27
             do ll=1,27
                if (abs(TransformationArray(kk,ll,jj,ii)).lt.1.d-12) then
                   TransformationArray(kk,ll,jj,ii)=0.d0
                end if
             end do
          end do
       end do
    end do
    deallocate(Transformation)
    deallocate(TransformationAux)
    deallocate(AllList)
    ! The memory used by the Fortran vectors must be assigned to
    ! the equivalent C pointers before finishing.
    cNequi=c_loc(Nequi(1))
    cList=c_loc(List(1,1))
    cALLEquiList=c_loc(ALLEquiList(1,1,1))
    cTransformationArray=c_loc(TransformationArray(1,1,1,1))
    cNIndependentBasis=c_loc(NIndependentBasis(1))
    cIndependentBasis=c_loc(IndependentBasis(1,1))
  end subroutine wedge

  ! Free the memory space used by the results of wedge().
  subroutine free_wedge(Allocsize,Nsymm,cNequi,cList,cALLEquiList,&
       cTransformationArray,cNIndependentBasis,&
       cIndependentBasis) bind(C,name="free_wedge")
    implicit none

    integer(kind=C_INT),value,intent(in) :: Allocsize,Nsymm
    type(C_PTR),value,intent(in) :: cNequi,cList,cALLEquiList,&
         cTransformationArray,cNIndependentBasis,cIndependentBasis

    integer(kind=C_INT),pointer :: Nequi(:),List(:,:),ALLEquiList(:,:,:)
    integer(kind=C_INT),pointer :: NIndependentBasis(:),IndependentBasis(:,:)
    real(kind=C_DOUBLE),pointer :: TransformationArray(:,:)

    call c_f_pointer(cNequi,Nequi,shape=[Allocsize])
    call c_f_pointer(cList,List,shape=[3,Allocsize])
    call c_f_pointer(cALLEquiList,ALLEquiList,shape=[3,Nsymm*6,Allocsize])
    call c_f_pointer(cTransformationArray,TransformationArray,&
         shape=[27,27,Nsymm*6,Allocsize])
    call c_f_pointer(cNIndependentBasis,NIndependentBasis,&
         shape=[Allocsize])
    call c_f_pointer(cIndependentBasis,IndependentBasis,&
         shape=[27,Allocsize])
  end subroutine free_wedge

  ! Each symmetry operation defines a mapping between atom indices in
  ! the supercell. This subroutine fills a matrix with those
  ! permutations.
  subroutine symmetry(Nsymm,Natoms,LatVec,Coord,ID_equi,&
       Ngrid1,Ngrid2,Ngrid3,Orth,Trans)
    implicit none

    integer(kind=C_INT),intent(in) :: Nsymm,Natoms,Ngrid1,Ngrid2,Ngrid3
    real(kind=C_DOUBLE),intent(in) :: LatVec(3,3),Coord(3,Natoms)
    real(kind=C_DOUBLE),intent(in) :: Orth(3,3,Nsymm),Trans(3,Nsymm)
    integer(kind=C_INT),intent(out) :: ID_equi(Nsymm,Natoms*Ngrid1*Ngrid2*Ngrid3)

    integer(kind=C_INT) :: Ind_species(Ngrid1*Ngrid2*Ngrid3*Natoms)
    integer(kind=C_INT) :: Ind_cell(3,Ngrid1*Ngrid2*Ngrid3*Natoms)
    integer(kind=C_INT) :: i,ispecies,isym,ispecies_sym,vec(3)
    real(kind=C_DOUBLE) :: Car(3),Car_sym(3,Nsymm)


    call Id2Ind(Ind_cell,Ind_species,Ngrid1,Ngrid2,Ngrid3,Natoms)
    do i=1,Natoms*Ngrid1*Ngrid2*Ngrid3
       vec=Ind_cell(:,i)
       ispecies=Ind_species(i)
       call Lattice2Car(Natoms,LatVec,Coord,vec,ispecies,Car)
       call symm(Nsymm,LatVec,Car,Car_sym,Orth,Trans)
       do isym=1,Nsymm
          call Car2Lattice(Natoms,LatVec,Coord,vec,ispecies_sym,Car_sym(:,isym))
          vec(1)=modulo(vec(1),Ngrid1)
          vec(2)=modulo(vec(2),Ngrid2)
          vec(3)=modulo(vec(3),Ngrid3)
          ID_equi(isym,i)=Ind2Id(vec,ispecies_sym,Ngrid1,Ngrid2,Natoms)
       end do
    end do
  end subroutine symmetry

  ! Apply a symmetry operation to a vector and return the result.
  subroutine symm(Nsymm,LatVec,r_in,r_out,Orth,Trans)
    implicit none

    integer(kind=C_INT),intent(in) :: Nsymm
    real(kind=C_DOUBLE),intent(in) :: LatVec(3,3),Orth(3,3,Nsymm),Trans(3,Nsymm)
    real(kind=C_DOUBLE),intent(in) :: r_in(3)
    real(kind=C_DOUBLE),intent(out) :: r_out(3,Nsymm)

    integer(kind=C_INT) :: ii
    real(kind=C_DOUBLE) :: dispCartesian(3)

    do ii=1,Nsymm
       dispCartesian=Trans(1,ii)*LatVec(:,1)+Trans(2,ii)*LatVec(:,2)+Trans(3,ii)*LatVec(:,3)
       r_out(:,ii)=matmul(Orth(:,:,ii),r_in)+dispCartesian
    end do
  end subroutine symm

  ! Return the unit cell and atom indices of an element of the
  ! supercell based on its Cartesian coordinates.
  subroutine Car2Lattice(Natoms,LatVec,Coord,Ind_cell,Ind_atom,Car)
    implicit none

    integer(kind=C_INT),intent(in) :: Natoms
    real(kind=C_DOUBLE),intent(in) :: LatVec(3,3),Coord(3,Natoms),Car(3)
    integer(kind=C_INT),intent(out) :: Ind_cell(3),Ind_atom

    real(kind=C_DOUBLE) :: displ(3),dist,tmp1(3,3),tmp2(3,Natoms)
    integer(kind=C_INT) :: i,piv(3)

    Ind_atom=0
    tmp1=LatVec
    do i=1,Natoms
       tmp2(:,i)=Car-Coord(:,i)
    end do
    call dgesv(3,Natoms,tmp1,3,piv,tmp2,3,i)
    do i=1,Natoms
       ind_cell=nint(tmp2(:,i))
       displ=Ind_cell(1)*LatVec(:,1)+Ind_cell(2)*LatVec(:,2)+&
            Ind_cell(3)*LatVec(:,3)-(Car-Coord(:,i))
       dist=displ(1)**2+displ(2)**2+displ(3)**2
       if (dist.lt.1.d-4) then
          Ind_atom=i
          exit
       end if
    end do
  end subroutine Car2Lattice

  ! Inverse of the previous subroutine: converts from atom+cell
  ! indices to Cartesian coordinates.
  subroutine Lattice2Car(Natom,LatVec,Coord,Ind_cell,Ind_atom,Car)
    implicit none

    integer(kind=C_INT),intent(in) :: Natom,Ind_cell(3),Ind_atom
    real(kind=C_DOUBLE),intent(in) :: LatVec(3,3)
    real(kind=C_DOUBLE),intent(in) :: Coord(3,Natom)
    real(kind=C_DOUBLE),intent(out) :: Car(3)

    integer(kind=C_INT) :: i

    do i=1,Natom
       if(Ind_atom.eq.i) then
          Car=Ind_cell(1)*LatVec(:,1)+Ind_cell(2)*LatVec(:,2)+Ind_cell(3)*LatVec(:,3)+Coord(:,i)
       end if
    end do
  end subroutine Lattice2Car

  ! Generate a mapping between unit cell+atom indices to atom indices
  ! in the supercell.
  subroutine Id2Ind(Ind_cell,Ind_species,Ngrid1,Ngrid2,Ngrid3,Nspecies)
    implicit none

    integer(kind=C_INT),intent(in) ::Ngrid1,Ngrid2,Ngrid3,Nspecies
    integer(kind=C_INT),intent(out) :: Ind_species(Ngrid1*Ngrid2*Ngrid3*Nspecies),&
         Ind_cell(3,Ngrid1*Ngrid2*Ngrid3*Nspecies)

    integer(kind=C_INT) :: ii,Ind_cellID(Ngrid1*Ngrid2*Ngrid3*Nspecies)

    do ii=1,Ngrid1*Ngrid2*Ngrid3*Nspecies
       Ind_species(ii)=modulo(ii-1,Nspecies)+1
       Ind_cellID(ii)=int((ii-1)/Nspecies)
    end do

    do ii=1,Ngrid1*Ngrid2*Ngrid3*Nspecies
       Ind_cell(3,ii)=int((Ind_cellID(ii))/(Ngrid1*Ngrid2))
       Ind_cell(2,ii)=int(modulo(Ind_cellID(ii),Ngrid1*Ngrid2)/Ngrid1)
       Ind_cell(1,ii)=modulo(Ind_cellID(ii),Ngrid1)
    end do
  end subroutine Id2Ind

  ! Split an atom index from the supercell into a set of cell+atom indices.
  integer(kind=C_INT) function Ind2Id(Ind_cell,Ind_species,Ngrid1,Ngrid2,Nspecies)
    implicit none

    integer(kind=C_INT),intent(in) ::Ind_cell(3),Ind_species,Ngrid1,Ngrid2,Nspecies

    Ind2Id=(Ind_cell(1)+(Ind_cell(2)+Ind_cell(3)*Ngrid2)*Ngrid1)*Nspecies+Ind_species
  end function Ind2Id

  ! Thin wrapper around gaussian() to ensure interoperability with C.
  subroutine cgaussian(ca,row,column,Ndependent,cb,NIndependent,cIndexIndependent)&
       bind(C,name="cgaussian")

    integer(kind=C_INT),value,intent(in) :: row,column
    integer(kind=C_INT),intent(out) :: Ndependent,NIndependent
    type(c_ptr),value,intent(in) :: ca,cb,cIndexIndependent

    integer(kind=C_INT),pointer :: IndexIndependent(:)
    real(kind=C_DOUBLE),pointer :: a(:,:),b(:,:)

    call c_f_pointer(ca,a,shape=[row,column])
    call c_f_pointer(cb,b,shape=[column,column])
    call c_f_pointer(cIndexIndependent,IndexIndependent,&
         shape=[column])
    call gaussian(a,row,column,Ndependent,b,NIndependent,IndexIndependent)
  end subroutine cgaussian

  ! Routine to perform Gaussian elimination. Used by wedge() to
  ! extract subsets of independent constants given overdetermined sets
  ! of linear constraints.
  subroutine gaussian(a,row,column,Ndependent,b,NIndependent,IndexIndependent)
    implicit none

    real(kind=C_DOUBLE),parameter :: EPS=1.d-10
    integer(kind=C_INT),value,intent(in) :: row,column
    integer(kind=C_INT),intent(out) :: Ndependent,&
         Nindependent,IndexIndependent(column)
    real(kind=C_DOUBLE),intent(inout) :: a(row,column)
    real(kind=C_DOUBLE),intent(out) :: b(column,column)

    integer(kind=C_INT) :: i,j,k,irow,Indexdependent(column)
    real(kind=C_DOUBLE) :: swap_ik(column)

    Nindependent=0
    Ndependent=0
    IndexIndependent=0
    swap_ik(:)=0.0d0

    irow=1
    do k=1,min(row,column)
       do i=1,row
          if(abs(a(i,k)).lt.EPS)a(i,k)=0.d0
       end do
       do i=irow+1,row
          if((abs(a(i,k))-abs(a(irow,k))).gt.eps) then
             do j=k,column
                swap_ik(j)=a(irow,j)
                a(irow,j)=a(i,j)
                a(i,j)=swap_ik(j)
             end do
          end if
       end do
       if(abs(a(irow,k)).gt.EPS) then
          Ndependent=Ndependent+1
          Indexdependent(Ndependent)=k
          do j=column,k,-1
             a(irow,j)=a(irow,j)/a(irow,k)
          end do
          if(irow.ge.2)then
             do i=1,irow-1
                do j=column,k,-1
                   a(i,j)=a(i,j)-a(irow,j)/a(irow,k)*a(i,k)
                end do
                a(i,k)=0.d0
             end do
          end if
          if(irow+1.le.row) then
             do i=irow+1,row
                do j=column,k,-1
                   a(i,j)=a(i,j)-a(irow,j)/a(irow,k)*a(i,k)
                end do
                a(i,k)=0.d0
             end do
             irow=irow+1
          end if
       else
          Nindependent=Nindependent+1
          IndexIndependent(Nindependent)=k
       end if
    end do
    b=0.d0
    if(Nindependent.gt.0) then
       do i=1,Ndependent
          do j=1,Nindependent
             b(Indexdependent(i),j)=-a(i,IndexIndependent(j))
          end do
       end do
       do j=1,Nindependent
          b(IndexIndependent(j),j)=1.d0
       end do
    end if
  end subroutine gaussian
end module thirdorder_fortran
