// mlir-hlo-opt %s \
// --hlo-canonicalize-scatter --legalize-mhlo-to-thlo \
// --hlo-legalize-to-linalg \
// --gml-tiling="tile-sizes=1 distribute=false op-name=linalg.generic" \
// --scalarize -cse --canonicalize |\
// mlir-hlo-opt \
// --empty-tensor-to-alloc-tensor \
// --hlo-one-shot-bufferize --canonicalize -cse \
// --convert-bufferization-to-memref \
// --gml-st-to-scf |\
// mlir-hlo-opt --buffer-results-to-out-params --convert-scf-to-cf \
// --generic-host-to-llvm -cse --canonicalize |\
// mlir-cpu-runner \
// -e main -entry-point-result=void \
// --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext \

func.func @max(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = mhlo.maximum %arg0, %arg1 : tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

func.func @main() {

  %1 = arith.constant dense<[[-1.0, 1.0, 0.0], [-0.0, -0.1, 0.1]]>
      : tensor<2x3xf32>
  %2 = arith.constant dense<[[-1.0, 1.0, 0.0], [-0.0, -0.1, 0.1]]>
      : tensor<2x3xf32>
  %3 = func.call @max(%1, %2) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %res_unranked = tensor.cast %3 : tensor<2x3xf32> to tensor<*xf32>
  func.call @printMemrefF32(%res_unranked) : (tensor<*xf32>) -> ()

  func.return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
