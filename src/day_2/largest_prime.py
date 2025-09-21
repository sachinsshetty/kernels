print("finding Largest Prime")

max_value = 100 

temp_array = []

count = 1

for value in range(1, max_value,2):
    temp_array.append(value)

print(len(temp_array))

original_array = temp_array

for value in original_array:
    
    for temp_value in range(value, max_value,2):

        temp_array.append(value)

    print(value)