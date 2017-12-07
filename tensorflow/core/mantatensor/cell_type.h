/*

class FlagGrid {
    const int* flagGrid;
    
    
    //! types of cells, in/outflow can be combined, e.g., TypeFluid|TypeInflow
    enum CellType {
            TypeNone     = 0,
            TypeFluid    = 1,
            TypeObstacle = 2,
            TypeEmpty    = 4,
            TypeInflow   = 8,
            TypeOutflow  = 16,
            TypeOpen     = 32,
            TypeStick    = 64,
            // internal use only, for fast marching
            TypeReserved = 256,
            // 2^10 - 2^14 reserved for moving obstacles
    };
    
    
  public:
    FlagGrid (const int*);
    
    //! check for different flag types
    inline bool isFluid(int idx) const { return flagGrid[idx] & TypeFluid; }
    inline bool isObstacle(int idx) const { return flagGrid[idx] & TypeObstacle; }
    inline bool isInflow(int idx) const { return flagGrid[idx] & TypeInflow; }
    inline bool isEmpty(int idx) const { return flagGrid[idx] & TypeEmpty; }
    inline bool isOutflow(int idx) const { return flagGrid[idx] & TypeOutflow; }
    inline bool isOpen(int idx) const { return flagGrid[idx] & TypeOpen; }
    inline bool isStick(int idx) const { return flagGrid[idx] & TypeStick; }
};

FlagGrid::FlagGrid (const int* flagGrid) {
  this->flagGrid = flagGrid;
}

*/

/*
struct Size {
  int width;
  int height;
  int depth;
} ;
*/

class FluidGrid {
    protected:
        //! types of cells, in/outflow can be combined, e.g., TypeFluid|TypeInflow
        enum CellType {
            TypeNone     = 0,
            TypeFluid    = 1,
            TypeObstacle = 2,
            TypeEmpty    = 4,
            TypeInflow   = 8,
            TypeOutflow  = 16,
            TypeOpen     = 32,
            TypeStick    = 64,
            // internal use only, for fast marching
            TypeReserved = 256,
            // 2^10 - 2^14 reserved for moving obstacles
        };
    
        const int* flags;
    
        int width;
        int height;
        int depth;
        int dim;
        
        int dimOffset[3];

        const float* vel;
        const float* den;
        
        inline int getDimOffset(int dim) { return dimOffset[dim];}
        
        inline bool isFluid(int idx)    const { return flags[idx] & TypeFluid; }
        inline bool isObstacle(int idx) const { return flags[idx] & TypeObstacle; }
        inline bool isInflow(int idx)   const { return flags[idx] & TypeInflow; }
        inline bool isEmpty(int idx)    const { return flags[idx] & TypeEmpty; }
        inline bool isOutflow(int idx)  const { return flags[idx] & TypeOutflow; }
        inline bool isOpen(int idx)     const { return flags[idx] & TypeOpen; }
        inline bool isStick(int idx)    const { return flags[idx] & TypeStick; }
        
        virtual bool kernelCondition(int) { return true; }
        virtual void kernelFunction(int, int, int);
        virtual void kernelIdentity(int);
        
    public:
        void init (const int*, const float*, const float*);
        void setSize(int, int, int);
        void setDimension(int);
        
        void runKernel(int);
        
};

void FluidGrid::init (const int* flags, const float* density, const float* velocity) {
    this->flags = flags;
    this->den = density;
    this->vel = velocity;
}

void FluidGrid::setSize (int width, int height, int depth) {
    this->width = width;
    this->height = height;
    this->depth = depth;
    
    dimOffset[0] = width*height;
    dimOffset[1] = width;
    dimOffset[2] = 1;
}

void FluidGrid::setDimension (int dimension) {
    this->dim = dimension;
}

void FluidGrid::runKernel (int border) {
    int idx = 0;
    int idxi = 0;

    for (int x = 0; x < width; x++) {
        bool xInside = x >= border && x < width - border;
        for (int y = 0; y < height; y++) {
            bool yInside = y >= border && y < height - border;
            for (int z = 0; z < depth; z++) {
                bool zInside = z > 0 && z < depth - 1;
                bool condition = kernelCondition(idx);
                for(int i = 0; i < dim; i++) {
                    if(xInside && yInside && zInside && condition) {
                        kernelFunction(idx, idxi, i);
                    } else {
                        kernelIdentity(idx);
                    }
                    idxi++;
                }
                idx++;
            }
        }
    }
}