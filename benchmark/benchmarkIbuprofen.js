import { run, bench, group, do_not_optimize } from 'mitata';
import { Matrix } from 'ml-matrix';

import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld.js';
import { randomSparseMatrix } from './utils/randomSparseMatrix.js';

/* eslint 
func-names: 0 
camelcase: 0
*/

const sizes = [8, 16, 32, 256, 512, 1024];
const densities = [0.125, 0.0625, 0.03125, 0.0039, 0.00197, 0.001]; //[0.01, 0.015, 0.02, 0.025, 0.03];

for (let i = 0; i < sizes.length; i++) {
  const size = sizes[i];
  const density = densities[i];
  const A = randomSparseMatrix(size, size, density);
  const B = randomSparseMatrix(size, size, density);
  let denseA = A.to2DArray();
  let denseB = B.to2DArray();
  const AOld = new SparseMatrixOld(denseA);
  const BOld = new SparseMatrixOld(denseB);
  // denseA = new Matrix(denseA);
  // denseB = new Matrix(denseB);

  group(`size:${size}-density:${density}`, () => {
    bench('mmulNew', () => {
      do_not_optimize(A.mmul(B));
    }); //.gc('inner');
    bench('mmul', () => {
      do_not_optimize(AOld.mmul(BOld));
    }); //.gc('inner');
    // bench('denseMatrix', () => {
    //   do_not_optimize(denseA.mmul(denseB));
    // }); //.gc('inner');
  });
}

await run({ silent: false });
