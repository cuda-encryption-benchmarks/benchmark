
#ifndef benchmark_run_H
#define benchmark_run_H


// Structure representing an abstraction for the data gathered 
// by a single run of the benchmark.
typedef struct {
        // The total time taken for the run.
        struct timespec time_elapsed;
        // The amount of global memory used for the block buffer.
        size_t buffer_size;
} benchmark_run_t;


#endif // benchmark_run_H
