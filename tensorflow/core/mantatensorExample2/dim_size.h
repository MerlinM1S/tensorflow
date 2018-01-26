#ifndef _DIM_SIZE_H
#define _DIM_SIZE_H
 
 class DimSize {
 public: 
    long batches;
    long width;
    long height;
    long depth; 
    long dim;
    
    DimSize() : batches(-1), width(-1), height(-1), depth(-1), dim(-1) { }
    
    DimSize(long _batches, long _width, long _height, long _depth, long _dim) : batches(_batches), width(_width), height(_height), depth(_depth), dim(_dim) { }
    
    long lengthOf(int dims) const {
        long result = 1;
        switch(dims) {
            case 5:
                result *= dim;
            case 4:
                result *= depth;
            case 3:
                result *= height;
            case 2:
                result *= width;
            case 1:
                result *= batches;
        }
        
        return result;
    }
    
    long batchSize(int dims) const {
        long result = 1;
        switch(dims) {
            case 5:
                result *= dim;
            case 4:
                result *= depth;
            case 3:
                result *= height;
            case 2:
                result *= width;
        }
        
        return result;
    }
    
    
    long batchToIndex(int dims, int batch) const {
        return batch * batchSize(dims);
    }
 };

#endif // _DIM_SIZE_H