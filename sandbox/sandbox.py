# Define a decorator function
def my_decorator(func):
    def wrapper():
        print("Before the function is called")
        result = func()  # Call the original function and store its result
        print("After the function is called")
        return result  # Return the result of the original function
    return wrapper

# Apply the decorator to a function
@my_decorator
def get_message():
    return


# Call the decorated function
message = get_message()
print("Message:", message)
