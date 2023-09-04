import numpy as np
from MulLayer import MulLayer
from AddLayer import AddLayer

applePrice: float = 100
appleCount: float = 2
orangePrice: float = 150
orangeCount: float = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward propagation
appleMoney = mul_apple_layer.forward(applePrice, appleCount)
orangeMoney = mul_orange_layer.forward(orangePrice, orangeCount)
totalMoney = add_apple_orange_layer.forward(appleMoney, orangeMoney)
totalWithTax = mul_tax_layer.forward(totalMoney, tax)
print(totalWithTax)

# back propagation
dPrice = 1
dTotalMoney, dTax = mul_tax_layer.backward(dPrice)
dAppleMoney, dOrnageMoney = add_apple_orange_layer.backward(dTotalMoney)
dOrnagePrice, dOrangeCount = mul_orange_layer.backward(dOrnageMoney)
dApplePrice, dAppleCount = mul_apple_layer.backward(dAppleMoney)
print(dApplePrice, dAppleCount, dOrnagePrice, dOrangeCount, dTax)
