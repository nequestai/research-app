def compute_sum(a, b, tolerance):
    n = 2
    partial_sum = a
    difference_sum = a

    while abs(difference_sum) > tolerance:
        ans = (a ** n) / (b ** (n - 1))

        new_partial_sum = partial_sum + ans
        difference_sum = abs(new_partial_sum - partial_sum)
        partial_sum = new_partial_sum

        print(n)

        print("Partial:", partial_sum)

        n += 1

        if n == 100000000:
            break

    return partial_sum


out = compute_sum(2.0, 1, 1e-16)
print(out)
