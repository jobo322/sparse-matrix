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
// const cardinalities = Array.from(
//   xSequentialFillFromStep({ from: 10, step: 5, size: 25 }),
// );

// const dimensions = Array.from(
//   xSequentialFillFromStep({ from: 700, step: 100, size: 13 }),
// );
const cardinalities = [120, 110];
const dimensions = [1024];

console.log(
  cardinalities
    .flatMap((e, i) => {
      return dimensions.map((d, di) => {
        return [d, e, (e / d ** 2).toExponential()];
      });
    })
    .sort((a, b) => a[0] - b[0])
    .join('\n'),
);
console.log(cardinalities.at(-1));

lineplot(() => {
  // bench('hibrid($cardinality,$dimension)', function* (ctx) {
  //   const cardinality = ctx.get('cardinality');
  //   const size = ctx.get('dimension');
  //   // Prepare matrices once
  //   let A = new SparseMatrix(randomMatrix(size, size, cardinality));
  //   let B = new SparseMatrix(randomMatrix(size, size, cardinality));

  //   // Benchmark the multiplication
  //   yield () => do_not_optimize(A.mmul(B));
  //   A = null;
  //   B = null;
  // })
  //   .gc('inner')
  //   .args('cardinality', cardinalities) //.range('size', 32, 1024, 2); //.args('size', sizes);
  //   .args('dimension', dimensions);

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

    const dummy = new Array(1000).fill(0);
    do_not_optimize(dummy);

    // Benchmark the multiplication
    yield () => {
      const result = A._mmulMediumDensity(B);
      do_not_optimize(result);
      return result;
    };

    // Explicit cleanup
    do_not_optimize(A);
    do_not_optimize(B);
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
function createSeededRandom(seed) {
  let state = seed;
  return function () {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}
// Run benchmarks and capture results
const results = await run({
  // Force GC between every benchmark
  gc: true,
  // More samples for statistical significance
  min_samples: 20,
  max_samples: 200,
  // Longer warmup to stabilize CPU state
  warmup_samples: 10,
  warmup_threshold: 100, // ms
  // Longer minimum time for stable measurements
  min_cpu_time: 2000, // 2 seconds minimum
  // Batch settings to reduce variance
  batch_samples: 5,
  batch_threshold: 10, // ms
  // Enable colors
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
  `benchmark-results5.json`,
  JSON.stringify(processedResults, null, 2),
);
