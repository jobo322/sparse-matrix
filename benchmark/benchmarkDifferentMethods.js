import { run, bench, lineplot, do_not_optimize } from 'mitata';
import { xSequentialFillFromStep } from 'ml-spectra-processing';

import { SparseMatrix } from '../src/index.js';

import { randomMatrix } from './utils/randomMatrix.js';
import { writeFile } from 'node:fs/promises';

/* eslint 
func-names: 0
camelcase: 0
*/
// Prepare matrices once
const cardinalities = Array.from(
  xSequentialFillFromStep({ from: 10, step: 10, size: 1 }),
);
const dimensions = Array.from(
  xSequentialFillFromStep({ from: 10, step: 25, size: 1 }),
);
lineplot(() => {
  // bench('mmulSmall($size)', function* (ctx) {
  //   const cardinality = ctx.get('cardinality');
  //   // Prepare matrices once
  //   const A = new SparseMatrix(randomMatrix(size, size, cardinality));
  //   const B = new SparseMatrix(randomMatrix(size, size, cardinality));
  //   // Benchmark the multiplication
  //   yield () => do_not_optimize(A._mmulSmall(B));
  //  A = null;
  //   B = null;
  // }).args('cardinality', sizes); //.range('size', 32, 1024, 2); // 16, 32, 64, 128, 256

  bench('low($cardinality,$dimension)', function* (ctx) {
    const cardinality = ctx.get('cardinality');
    const size = ctx.get('dimension');
    // Prepare matrices once
    let A = new SparseMatrix(randomMatrix(size, size, cardinality));
    let B = new SparseMatrix(randomMatrix(size, size, cardinality));

    // Benchmark the multiplication
    yield () => do_not_optimize(A._mmulLowDensity(B));
    A = null;
    B = null;
  })
    .gc('inner')
    .args('cardinality', cardinalities) //.range('size', 32, 1024, 2); //.args('size', sizes);
    .args('dimension', dimensions);

  bench('medium($cardinality,$dimension)', function* (ctx) {
    const cardinality = ctx.get('cardinality');
    const size = ctx.get('dimension');
    // Prepare matrices once
    let A = new SparseMatrix(randomMatrix(size, size, cardinality));
    let B = new SparseMatrix(randomMatrix(size, size, cardinality));

    // Benchmark the multiplication
    yield () => do_not_optimize(A._mmulMediumDensity(B));
    A = null;
    B = null;
  })
    .gc('inner')
    .args('cardinality', cardinalities) //.range('size', 32, 1024, 2); //.args('size', sizes);
    .args('dimension', dimensions);

  // bench('mmul($cardinality)', function* (ctx) {
  //   const cardinality = ctx.get('cardinality');
  //   // Prepare matrices once
  //   const A = new SparseMatrix(randomMatrix(size, size, cardinality));
  //   const B = new SparseMatrix(randomMatrix(size, size, cardinality));

  //   // Benchmark the multiplication
  //   yield () => do_not_optimize(A.mmul(B));
  // }).args('cardinality', sizes);
});

// Run benchmarks and capture results
const results = await run({
  // Enable garbage collection between benchmarks
  gc: true,
  // More conservative sampling to allow GC to work
  min_samples: 10,
  max_samples: 100,
  // Longer minimum CPU time to get stable results
  min_cpu_time: 1000, // 1 second minimum
  // Enable colors for better output readability
  colors: true,
});

// Process and store results
const processedResults = [];

for (const benchmark of results.benchmarks) {
  for (const run of benchmark.runs) {
    if (run.stats) {
      processedResults.push({
        name: benchmark.alias,
        cardinality: run.args.cardinality,
        dimension: run.args.dimension,
        avg: run.stats.avg,
        min: run.stats.min,
        max: run.stats.max,
        p50: run.stats.p50,
        p75: run.stats.p75,
        p99: run.stats.p99,
        samples: run.stats.samples.length,
        ticks: run.stats.ticks,
      });
    }
  }
}

// Save results to JSON file
await writeFile(
  `benchmark-results.json`,
  JSON.stringify(processedResults, null, 2),
);
