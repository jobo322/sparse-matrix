import { run, bench, lineplot, do_not_optimize } from 'mitata';
import { xSequentialFillFromStep } from 'ml-spectra-processing';

import { SparseMatrix } from '../src/index.js';

import { randomMatrix } from './utils/randomMatrix.js';
import { writeFile } from 'node:fs/promises';

const size = 256; // Fixed density for this comparison;

/* eslint 
func-names: 0
camelcase: 0
*/
// Prepare matrices once
const sizes = Array.from(
  xSequentialFillFromStep({ from: 10, step: 5, size: 20 }),
);
lineplot(() => {
  bench('mmulSmall($size)', function* (ctx) {
    const cardinality = ctx.get('cardinality');
    // Prepare matrices once
    const A = new SparseMatrix(randomMatrix(size, size, cardinality));
    const B = new SparseMatrix(randomMatrix(size, size, cardinality));
    // Benchmark the multiplication
    yield () => do_not_optimize(A.mmul(B));
  }).args('cardinality', sizes); //.range('size', 32, 1024, 2); // 16, 32, 64, 128, 256

  bench('mmulLowDensity()($cardinality)', function* (ctx) {
    const cardinality = ctx.get('cardinality');
    // Prepare matrices once
    const A = new SparseMatrix(randomMatrix(size, size, cardinality));
    const B = new SparseMatrix(randomMatrix(size, size, cardinality));

    // Benchmark the multiplication
    yield () => do_not_optimize(A._mmulMediumDensity(B));
  }).args('cardinality', sizes); //.range('size', 32, 1024, 2); //.args('size', sizes);

  // bench('CSC($cardinality)', function* (ctx) {
  //   const cardinality = ctx.get('cardinality');
  //   // Prepare matrices once
  //   const A = new SparseMatrix(randomMatrix(size, size, cardinality));
  //   const B = new SparseMatrix(randomMatrix(size, size, cardinality));

  //   // Benchmark the multiplication
  //   yield () => do_not_optimize(A.mmul(B));
  // }).args('cardinality', sizes);
});

// Run benchmarks and capture results
const results = await run();

// Process and store results
const processedResults = [];

for (const benchmark of results.benchmarks) {
  for (const run of benchmark.runs) {
    if (run.stats) {
      processedResults.push({
        name: benchmark.alias,
        size: run.args.size,
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
  `benchmark-results-${size}.json`,
  JSON.stringify(processedResults, null, 2),
);
