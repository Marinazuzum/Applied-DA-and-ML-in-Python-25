import numpy as np
import random

# Task 1: Replace negative numbers with 0 in the database
def task1(database):
    # Loop through the database and replace negative numbers with 0
    for i in range(len(database)):
        if database[i] < 0:
            database[i] = 0
    return database

# Task 2: Check if a number is prime
def task2(number):
    # A prime number is only divisible by 1 and itself
    if number <= 1:
        print(f"{number} is not a prime number.")
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            print(f"{number} is not a prime number.")
            return False
    print(f"{number} is a prime number.")
    return True

# Task 3: Create an array with random numbers and determine the relationships
def task3():
    # Generate three random numbers between 0 and 2
    numbers = [random.randint(0, 2) for _ in range(3)]
    
    print(f"Generated numbers: {numbers}")
    
    # Check different conditions
    if numbers[0] == numbers[1] == numbers[2]:
        print("All numbers are the same.")
    elif numbers[0] == numbers[2]:
        print("The first and the last number are the same.")
    elif numbers[0] == numbers[1]:
        print("The first two numbers are the same.")
    elif numbers[1] == numbers[2]:
        print("The last two numbers are the same.")
    else:
        print("All numbers are different.")

if __name__ == "__main__":
    print('Task 1:')
    database = np.array([-1, 2, 6, -5, 0, -3, 8, 0.01, 5, -3, 7, 8, 5, 3, 2, 0, -0.11, 4.45])
    print(task1(database))
    
    print('\n\nTask 2:')
    numbers2check = (9, 5, 10, 17, 563, 1998, 1973)
    result = []
    for number in numbers2check:
        result.append(task2(number))
    
    print('\n\nTask 3:')
    task3()
