
class Employee:
    def __init__(self,emp_id,emp_name,emp_hourly_wage):
        self.emp_id = emp_id
        self.emp_name = emp_name
        self.emp_hourly_wage = emp_hourly_wage

    def details(self):
        print("Id : " , self.emp_id)
        print("Name : " , self.emp_name)
        print("Hourly wage : ",self.emp_hourly_wage,"$")
        print("--------------------------------------------------")


# employees are saved into a dict of employee with the emp_id as the key and the information of the employee is saved as the value 

employees = {}

def create_employee(emp_id,emp_name,emp_hourly_wage):
    # Check if the Employee is already created.
    # make sure it is not created before.
    # if it is created just say the employee is already exisiting.
    if emp_id not in employees:
        new_employee = Employee(emp_id,emp_name,emp_hourly_wage)
        employees[emp_id] = new_employee
        print("Saved",emp_name,"as a new employee")
    else:
        print("Empoyee",emp_id,"is already Existing ")
    
def show_employees():
    for emp_id in employees:
        employee = employees[emp_id]
        employee.details()


def update_employee(emp_id,new_name,new_hourly_wage):
    if emp_id in employees:
        employee = employees[emp_id]
        if new_name and new_hourly_wage:
            employee.emp_name = new_name
            employee.emp_hourly_wage = new_hourly_wage
            print("Employee with",emp_id,"is updated.")
        else:
            print("Invalid arguments name and hourly wages has to be mentioned.")
    else:
        print("Employee with",emp_id,"does't exist")


def delete_employee(emp_id):
    if emp_id in employees:
        del employees[emp_id]
        print("Successfully deleted the employee",emp_id)
    else:
        print("Employee id : ",emp_id,"is not found in the registry.")

# days 25 per each day 8 hrs hourly are given.

def emp_salary(emp_id):
    if emp_id in employees:
        employee = employees[emp_id]
        salary = (25 * 8) * employee.emp_hourly_wage
        return salary
    else:
        raise Exception("Employee Not Found")


# this is changed here.
# this is changed here.
# this is changed here.









# Using the Functions for managing the employees.

create_employee("Emp1","Basha Vali",300)
create_employee("Emp2","Rithika",400)
create_employee("Emp3","Radha",350)


update_employee("Emp2","Rithika Singh",420)

delete_employee("Emp3")
print(emp_salary("Emp2"))
show_employees()



    
