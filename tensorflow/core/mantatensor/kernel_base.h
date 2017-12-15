#ifndef KERNEL_BASE_H_
#define KERNEL_BASE_H_

#include "fluid_grid_functor.h"

class KernelBase {
    protected:
        const FluidGridFunctor& fluidGridFunctor;

    private:
        const int border;

    protected:

        virtual void kernelFunction(int i_bxyz) = 0;
        virtual void kernelIdentityDim(int i_bxyzd) = 0;

        inline int get_i_bxyzd(int i_bxyz, int i_d) { return fluidGridFunctor.getDim()*i_bxyz + i_d; }


        inline void kernelIdentity(int i_bxyz) {
            for(int d = 0; d < fluidGridFunctor.getDim(); d++) {
                kernelIdentityDim(get_i_bxyzd(i_bxyz, d));
            }
        }

    public:
        KernelBase(const FluidGridFunctor& fluidGridFunctor, int border) : fluidGridFunctor(fluidGridFunctor), border(border) { }


        void run() {
            for (int b = 0; b < fluidGridFunctor.getBatches(); b++) {
                int i_b = b * fluidGridFunctor.getWidth() * fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth();

                for (int x = 0; x < fluidGridFunctor.getWidth(); x++) {
                    int i_bx = i_b + x*fluidGridFunctor.getHeight() * fluidGridFunctor.getDepth();
                    bool xInside = x >= border && x < fluidGridFunctor.getWidth() - border;
                    for (int y = 0; y < fluidGridFunctor.getHeight(); y++) {
                        int i_bxy = i_bx + y * fluidGridFunctor.getDepth();
                        bool yInside = y >= border && y < fluidGridFunctor.getHeight() - border;
                        for (int z = 0; z < fluidGridFunctor.getDepth(); z++) {
                            int i_bxyz = i_bxy + z;
                            bool zInside = z >= border && z < fluidGridFunctor.getDepth() - border;

                            if(xInside && yInside && zInside) {
                                kernelFunction(i_bxyz);
                            } else {
                                kernelIdentity(i_bxyz);
                            }
                        }
                    }
                }
            }
        }
}; 

#endif // KERNEL_BASE_H_