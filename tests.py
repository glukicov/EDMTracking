'''
Define some unit test and integration tests
'''

def main():
    test_sum()

    print("Everything passed")

def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

if __name__ == "__main__":
    main()
