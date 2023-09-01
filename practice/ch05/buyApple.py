from MulLayer import MulLayer

apple_price = 100
apple_count = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward propagation
apple_price = mul_apple_layer.forward(apple_price, apple_count)
total_price = mul_tax_layer.forward(apple_price, tax)
print(total_price)

# back propagation
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_count = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_count, dtax)
