"""""
test
"""""
import keyword
a = "RUH"
c = 3

mo, se, xy = "Sedky", 2, 0.779


print(a)
print(c)

print(c is len(a))
print(c != len(a))

x=keyword.kwlist
print(x)

print(mo)
print(se)
print(xy)


##########################################################################################

int_num = 10
float_num = 20.90
str_val="thiqah"

add = int_num+float_num
sub = int_num-float_num
multi = int_num*float_num
str_concat = str_val+str(int_num)
div = int_num/float_num
exponential = 10 ** 20
remainder = 500 % 6
string_with_quotes = "Hello 'Sedky' , How are you ?"
string_with_ignore_quotes = "Hello \"Sedky\" , How are you ?"
tr = True
fls = False

first = 'PYTHon'[0:4]
second = 'is' [0:2]
third = 'functional'
third = str(len(third))
fourth = 'Programming'

replaceable_str = "JAVA Selenium is not ONLY AUTOMATION"

print('add=', add)
print("sub=", sub)
print("multi=", multi)
print("str_concat=", str_concat)
print("div=", div)
print("exponential=", exponential)
print("remainder=", remainder)
print(tr)
print(fls)
print(bool("1"=='1'),bool(0))
print(bool("python"))
print(string_with_quotes)
print(string_with_ignore_quotes)
print("first=", first)
print(first.lower()+" "+second.upper()+" "+third+" "+fourth)
print(replaceable_str)
print(replaceable_str[::-1])
print("Test for Python %s and %s"%(replaceable_str, remainder))
print(replaceable_str.split())
print(replaceable_str[:2])

print(replaceable_str[0:4].replace('JAVA', 'Python')+" "+replaceable_str[5:16] +
      replaceable_str[17:20].replace(replaceable_str[17:20], '')+""+replaceable_str[25:]+" as well")


