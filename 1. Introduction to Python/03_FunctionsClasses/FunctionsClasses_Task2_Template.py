def square_numbers(n, memo=None):
    # Initialize memoization dictionary if not provided
    if memo is None:
        memo = {}
    
    # Base cases: we know that the sequence starts with a_0 = 0, a_1 = 1, and a_2 = 4
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 4

    # If the value is already computed and stored in memo, return it
    if n in memo:
        return memo[n]

    # Recursive case: Use the given recursive relation
    result = 3 * square_numbers(n-1, memo) - 3 * square_numbers(n-2, memo) + square_numbers(n-3, memo)
    
    # Store the result in memo to avoid redundant calculations
    memo[n] = result
    return result

# Example usage:
if __name__ == "__main__":
    for i in range(10):
        print('Number:\t', i, '\t\tSquared number:\t', square_numbers(i))




