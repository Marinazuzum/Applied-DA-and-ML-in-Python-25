class Student:
    def __init__(self, first_name, last_name, grades, bonus=''):
        self.first_name = first_name
        self.last_name = last_name
        self.grades = grades
        self.bonus = bonus
        self.email = ''
        self.avg_grade = 0.0
        self.passed = False
    
    def compute_grades(self):
        # Check if any grade is higher than 4.0, student failed
        if any(grade > 4.0 for grade in self.grades):
            self.avg_grade = 5.0
            self.passed = False
        else:
            # Calculate the average grade
            self.avg_grade = round(sum(self.grades) / len(self.grades), 1)
            self.passed = self.avg_grade <= 4.0
            
            # Add bonus if applicable
            if self.bonus == 'yes':
                self.avg_grade = round(self.avg_grade + 0.3, 1)
    
    def create_email(self):
        # Create the email based on the format FirstLetterOfFirstName.LastName@tum.de
        self.email = f"{self.first_name[0]}.{self.last_name}@tum.de"
    
    def create_student_file(self):
        # Create a file with the student's data
        file_name = f"1. Introduction to Python/05_StudentDatabase/Grades_{self.first_name}_{self.last_name}.txt"
        with open(file_name, 'w') as f:
            f.write(f"Name:\t{self.first_name} {self.last_name}\n")
            #f.write(f"Grades: {', '.join(map(str, self.grades))}\n")
            f.write(f"Email:\t{self.email}\n")
            f.write(f"Average Grade:\t{self.avg_grade}\n\n")
            if self.passed:
                f.write(f"{self.first_name} passed the course.")
            else:  
                f.write(f"{self.first_name} not passed the course.")
    
def evaluate_database():
    students = []
    total_grades = 0
    passed_count = 0
    best_student = None
    best_avg_grade = 10
    
    # Read the student data from the Students.txt file
    #with open("05_InputOutput/PythonWikipedia_reversed.txt", "r") as infile:
    try:
        with open("1. Introduction to Python/05_StudentDatabase/Students.txt", "r") as file:
            for line in file:
                # Split each line into components
                parts = line.strip().split(',')
                # Handle cases where there might be leading/trailing spaces
                first_name, last_name = parts[0].strip().split()
                grades = list(map(float, parts[1:-1]))  # Convert grades to floats
                bonus = parts[-1].strip()  # Bonus is the last part (yes/no)
                
                # Create a Student object and compute grades
                student = Student(first_name, last_name, grades, bonus)
                student.compute_grades()
                student.create_email()
                student.create_student_file()
                
                # Append the student to the list
                students.append(student)
                
                # Update the best student
                if student.avg_grade < best_avg_grade:
                    best_avg_grade = student.avg_grade
                    best_student = student
                
                # Update statistics
                total_grades += student.avg_grade
                if student.passed:
                    passed_count += 1
    
        # Calculate course statistics
        average_grade = total_grades / len(students) if students else 0
        pass_percentage = (passed_count / len(students)) * 100 if students else 0
        
        # Print the result
        print(f"The average grade of the course was {average_grade:.2f}")
        print(f"{pass_percentage:.1f}% of the students passed the course")
        print(f"The best student was {best_student.first_name} {best_student.last_name} with an average grade of {best_avg_grade}")
    
    except FileNotFoundError:
        print("The file Students.txt was not found.")

# Main function to evaluate the database
if __name__ == "__main__":
    evaluate_database()
