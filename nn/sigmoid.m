# Sigmoid function
function a = sigmoid(val)
  a = 1 ./ (1 .+ exp(val));
endfunction