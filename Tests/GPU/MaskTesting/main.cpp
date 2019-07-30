#include <cuda_device_runtime_api.h>

#include <iostream>
#include <AMReX.H>
#include <AMReX_Print.H>

#include <AMReX_Geometry.H>
#include <AMReX_ArrayLim.H>
#include <AMReX_Vector.H>
#include <AMReX_IntVect.H>
#include <AMReX_BaseFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Gpu.H>

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        int n_cell, max_grid_size, nsteps, plot_int;
        Vector<int> is_periodic(AMREX_SPACEDIM,1);  // periodic in all direction by default

        {
            ParmParse pp;

            pp.get("n_cell",n_cell);
            pp.get("max_grid_size",max_grid_size);
            pp.queryarr("is_periodic", is_periodic);

            plot_int = -1;
            pp.query("plot_int",plot_int);

            nsteps = 10;
            pp.query("nsteps",nsteps);
        }

        // make BoxArray
        BoxArray ba;
        BoxArray baB;

        Box domain({0,0,0}, {n_cell-1, n_cell-1, n_cell-1});
        Box subdomain({0,0,1}, {n_cell-1, n_cell-1, n_cell-2});
        ba.define(domain);
        ba.maxSize(max_grid_size);

        int Nghost = 3; 
        int Ncomp  = 2;
        DistributionMapping dm(ba);

        amrex::Print() << std::endl << std::endl;

        iMultiFab mask(ba, dm, Ncomp, Nghost, MFInfo(), DefaultFabFactory<IArrayBox>());
        iMultiFab maskA(ba, dm, Ncomp, Nghost, MFInfo(), DefaultFabFactory<IArrayBox>());
        iMultiFab maskB(ba, dm, Ncomp, Nghost, MFInfo(), DefaultFabFactory<IArrayBox>());

        // ===============

        {
            iMultiFab first(ba, dm, Ncomp, Nghost, MFInfo(), DefaultFabFactory<IArrayBox>());
            iMultiFab firstA(ba, dm, Ncomp, Nghost, MFInfo(), DefaultFabFactory<IArrayBox>());
            iMultiFab firstB(ba, dm, Ncomp, Nghost, MFInfo(), DefaultFabFactory<IArrayBox>());

            first.BuildMask(subdomain, Periodicity(IntVect{1,1,1}), 1, 2, 3, 4);
            firstA.BuildMaskA(subdomain, Periodicity(IntVect{1,1,1}), 1, 2, 3, 4);
            firstB.BuildMaskB(subdomain, Periodicity(IntVect{1,1,1}), 1, 2, 3, 4);
        }

        // ===============

        {
            BL_PROFILE("MASK");
            mask.BuildMask(subdomain, Periodicity(IntVect{1,1,1}), 1, 2, 3, 4);
        }
        {
            BL_PROFILE("MASKA");
            maskA.BuildMaskA(subdomain, Periodicity(IntVect{1,1,1}), 1, 2, 3, 4);
        }
        {
            BL_PROFILE("MASKC");
            maskB.BuildMaskC(subdomain, Periodicity(IntVect{1,1,1}), 1, 2, 3, 4);
        }

        // ===============


        // Check answer
        {
            for (MFIter mfi(mask); mfi.isValid(); ++mfi)
            {
                const Box& gbx = mfi.growntilebox();
                auto m = mask.array(mfi);
                auto mA = maskA.array(mfi);
                auto mB = maskB.array(mfi);

                AMREX_HOST_DEVICE_FOR_4D(gbx, Ncomp, i, j, k, n,
                {
                    if (m(i,j,k,n) != mA(i,j,k,n))
                    {
                        printf("(A) %i %i %i: %i != %i\n", i, j, k, m(i,j,k,n), mA(i,j,k,n));
                    }

                    if (m(i,j,k,n) != mB(i,j,k,n))
                    {
                        printf("(C) %i %i %i: %i != %i\n", i, j, k, m(i,j,k,n), mB(i,j,k,n));
                    }

                });
            }
        }
    }
    amrex::Finalize();
}
