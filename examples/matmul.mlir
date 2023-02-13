func.func @matmul(%1: tensor<3000x256xf32>, %2: tensor<256x1024xf32>) -> tensor<3000x1024xf32> {
  %ret = "mhlo.dot"(%1, %2) : (tensor<3000x256xf32>, tensor<256x1024xf32>) -> tensor<3000x1024xf32>
  return %ret : tensor<3000x1024xf32>
}