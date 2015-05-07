# Feed forward 
function a = feedfwd(X, Theta)
  a = sigmoid(X * Theta');
endfunction