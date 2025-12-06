module {
  func.func @fill_square(%size: index) -> tensor<?x?xf32> {
    %result = tensor.generate %size, %size {
      ^bb0(%i: index, %j: index):
        // Step 1: integer multiply on indices
        %prod = arith.muli %i, %j : index
        // Step 2: cast index -> i64 (integer)
        %prod_i64 = arith.index_cast %prod : index to i64
        // Step 3: cast i64 -> f32 (float)
        %prod_f32 = arith.sitofp %prod_i64 : i64 to f32
        tensor.yield %prod_f32 : f32
    } : tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
}
