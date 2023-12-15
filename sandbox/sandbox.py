from scipy.optimize import minimize


# Define your objective function
def objective_function(args):
    return args[0]*2 + args[0] - args[0]


target = 10


def obj_func_final(args):
    return abs(objective_function(args) - target)


# Specify the optimization bounds for each argument
bounds = [(1, 20)]  # Example bounds, adjust according to your problem

# Run the optimization
result = minimize(obj_func_final, [2], bounds=bounds)

# Print the optimized arguments
optimized_arguments = result.x
print("Optimized Arguments:", optimized_arguments)

# Print the optimized function value
optimized_value = result.fun
print("Optimized Difference Value:", optimized_value)

# Print the result of the function
function_result = optimized_value + target
print("Funtion Result:", function_result)

# Prove
prove = objective_function(optimized_arguments)
print("Evidence:", prove)

tolerance = 1.0e-6
if optimized_value < tolerance:
    print("Status: Succeed")

print(objective_function([6]))