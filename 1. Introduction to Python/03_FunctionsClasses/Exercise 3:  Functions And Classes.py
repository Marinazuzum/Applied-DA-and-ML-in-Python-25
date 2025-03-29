
#Task1:
#• Start from scratch and import numpy
# Define two classes for geometric objects: Circle and Rectangle with Radius and Width
#and Length parameters, respectively. Parameters should be initialized to 5, if no value
#is given, when creating an object of the class
#• For each class implement the functions compute_area() and compute_perimeter(),
#which should return the computed value.
#• Define the function generate_objects(number) which is called from main() with
#number=10 and gets the number of objects as input and returns a list of number of
#randomly generated geometric objects with random parameters (int, range 1 to 10)
#• Calculate and print the mean area and the sum of the perimeters of all objects
import numpy as np
import random

# Define the Circle class
class Circle:
    def __init__(self, radius=5):
        self.radius = radius

    def compute_area(self):
        return np.pi * self.radius ** 2

    def compute_perimeter(self):
        return 2 * np.pi * self.radius

# Define the Rectangle class
class Rectangle:
    def __init__(self, width=5, length=5):
        self.width = width
        self.length = length

    def compute_area(self):
        return self.width * self.length

    def compute_perimeter(self):
        return 2 * (self.width + self.length)

# Function to generate random objects
def generate_objects(number):
    objects = []
    for _ in range(number):
        # Randomly choose between Circle or Rectangle
        if random.choice([True, False]):
            radius = random.randint(1, 10)
            objects.append(Circle(radius))
        else:
            width = random.randint(1, 10)
            length = random.randint(1, 10)
            objects.append(Rectangle(width, length))
    return objects

# Main function to calculate the mean area and sum of perimeters
def main():
    number = 10  # Given in the task
    objects = generate_objects(number)
    
    total_area = 0
    total_perimeter = 0
    
    for obj in objects:
        total_area += obj.compute_area()
        total_perimeter += obj.compute_perimeter()

    mean_area = total_area / number
    print(f"Mean area of all objects: {mean_area:.2f}")
    print(f"Sum of perimeters of all objects: {total_perimeter:.2f}")

# Call main function
main()

