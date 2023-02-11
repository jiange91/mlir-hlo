// mlir-hlo-opt %s \
// --legalize-mhlo-to-thlo="enable-experimental=true" \
// --gml-tiling="tile-sizes=1,1 distribute=false op-name=thlo.concatenate" \
// --scalarize -cse --canonicalize |\
// mlir-hlo-opt \
// --empty-tensor-to-alloc-tensor \
// --hlo-one-shot-bufferize --canonicalize -cse \
// --convert-bufferization-to-memref \
// --gml-st-to-scf --buffer-results-to-out-params --convert-scf-to-cf \
// --generic-host-to-llvm -cse --canonicalize | \
// mlir-cpu-runner \
// -e main -entry-point-result=void \
// --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext \
// | FileCheck %s

func.func @concat(%a: tensor<2x2xf32>, %b: tensor<2x3xf32>, %c: tensor<2x?xf32>)
    -> tensor<2x?xf32> {
  %concat = "mhlo.concatenate"(%a, %b, %c) { dimension = 1 }
      : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  func.return %concat : tensor<2x?xf32>
}

func.func @main() {
  %test_arg_a = arith.constant dense<[[1.11, 1.12], [1.21, 1.22]]> : tensor<2x2xf32>
  %test_arg_a_ = tensor.cast %test_arg_a : tensor<2x2xf32> to tensor<2x2xf32>
  %test_arg_b = arith.constant dense<[[2.11, 212., 2.13], [2.21, 2.22, 2.23]]> : tensor<2x3xf32>
  %test_arg_b_ = tensor.cast %test_arg_b : tensor<2x3xf32> to tensor<2x3xf32>
  %test_arg_c = arith.constant dense<[[], []]> : tensor<2x0xf32>
  %test_arg_c_ = tensor.cast %test_arg_c : tensor<2x0xf32> to tensor<2x?xf32>
  %test_concat = func.call @concat(%test_arg_a_, %test_arg_b_, %test_arg_c_)
      : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>

  // CHECK: rank = 2
  // CHECK: offset = 0
  // CHECK: sizes = [2, 5]
  // CHECK: strides = [5, 1]
  // CHECK: data =
  // CHECK:   1.11, 1.12, 2.11, 212, 2.13
  // CHECK:   1.21, 1.22, 2.21, 2.22, 2.23
  %test_concat_unranked = tensor.cast %test_concat
      : tensor<2x?xf32> to tensor<*xf32>
  func.call @printMemrefF32(%test_concat_unranked) : (tensor<*xf32>) -> ()
  func.return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
