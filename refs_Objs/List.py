Programming_Languages = ["JAVA", "Python", ".Net"]
numbers = [1, 2, 3.0, 0.6, 0.9, 1000000.2]


if len(numbers) > 4:
    numbers[len(numbers)-2] = 0.36
    numbers.append("text")
    numbers.insert(3, "new interfere")

Programming_Languages.remove(Programming_Languages[2])

for num in numbers:
    if numbers[num].isdigit():
       numbers.pop(num)


print(Programming_Languages)
print(Programming_Languages[0])
print(numbers)
print(numbers.index("new interfere"))
print(numbers.pop(len(numbers)-1), numbers)
print(numbers[0:2])
print("latest list is", numbers)
print((numbers.remove(numbers[3])).sort())
