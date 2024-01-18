import sys

def main():
    # Check if an argument is passed
    if len(sys.argv) != 2:
        print("Usage: python test.py <number>")
        sys.exit(1)

    # Parse the argument
    try:
        x = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid integer.")
        sys.exit(1)

    # Perform a simple operation, like squaring the number
    result = x ** 2

    # Print the results
    print(f"Received number: {x}")
    print(f"Square of the number: {result}")

if __name__ == "__main__":
    main()
