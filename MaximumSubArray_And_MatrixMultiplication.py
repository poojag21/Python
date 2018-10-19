# -*- coding: utf-8 -*-

from numpy import asarray
import numpy as numpy
import sys as sys


ENERGY_LEVEL = [100, 113, 110, 85, 105, 102, 86,
                63, 81, 101, 94, 106, 101, 79, 94, 90, 97]
ENERGY_LEVEL_2 = [110, 109, 107, 104, 100]
a = [[0, 1], [1, 1]]
k = 3
error_msg = 'Input is not valid!'


# ==============================================================

# The brute force method to solve first problem
def find_significant_energy_increase_brute(A):
    """
    Return a tuple (i,j) where A[i:j] is the
    most significant energy increase period.
    time complexity = O(n^2)
    """

    a = change_in_energy(A)
    n = len(a)
    max_till_now = 0
    for i in range(0, n):
        sum_till_now = 0
        for j in range(i, n):
            sum_till_now += a[j]
            if sum_till_now >= max_till_now:
                start_index = i
                end_index = j
                max_till_now = sum_till_now
    if start_index == 0 and end_index == 0:
        return 0, 1
    else:
        return (start_index - 1), end_index


# Method to calculate the change in Energy
def change_in_energy(A):
    n = len(A)
    a = numpy.zeros([n])
    for i in range(0, n-1):
        a[i+1] = A[i+1] - A[i]
    a[0] = 0
    return asarray(a)


# ==============================================================


# The recursive method to solve first problem
def find_significant_energy_increase_recursive(A):
    """
    Return a tuple (i,j) where A[i:j] is the
    most significant energy increase period.
    time complexity = O (n logn)
    """
    A = change_in_energy(A)
    low = 0
    high = len(A)-1
    maximum_change_in_energy, start_index, end_index = find_max_significant_energy_change(A, low, high)
    if start_index == 0 and end_index == 0:
        return 0, 1
    else:
        return start_index - 1, end_index


# To calculate maximum change in energy among right, left and the crossing sub-array
def find_max_significant_energy_change(A, low, high):
    mid = (high+low)//2
    if high == low:
        max_significant_energy = A[low]
        return max_significant_energy, low, high
    else:
        left_sub_array_sum, left_low, left_high = find_max_significant_energy_change(A, low, mid)
        right_sub_array_sum, right_low, right_high = find_max_significant_energy_change(A, mid+1, high)
        crossing_sub_array_sum, crossing_low, crossing_high = maximum_crossing_sub_array(A, low, mid, high)

        if max(left_sub_array_sum, right_sub_array_sum,
               crossing_sub_array_sum) == crossing_sub_array_sum:
            return crossing_sub_array_sum, crossing_low, crossing_high
        elif max(left_sub_array_sum, right_sub_array_sum,
                 crossing_sub_array_sum) == right_sub_array_sum:
            return right_sub_array_sum, right_low, right_high
        elif max(left_sub_array_sum, right_sub_array_sum,
                 crossing_sub_array_sum) == left_sub_array_sum:
            return left_sub_array_sum, left_low, left_high


# Method to find the maximum crossing sub-array
def maximum_crossing_sub_array(A, low, mid, high):
    max_till_now_left = - sys.maxsize - 1
    max_till_now_right = - sys.maxsize - 1
    sum_till_now_left = 0
    sum_till_now_right = 0
    start_index = 0
    end_index = 0

    for i in range(mid, low-1, -1):
        sum_till_now_left += A[i]

        if max_till_now_left < sum_till_now_left:
            max_till_now_left = sum_till_now_left
            start_index = i

    for j in range(mid+1, high+1):
        sum_till_now_right += A[j]
        if max_till_now_right < sum_till_now_right:
            max_till_now_right = sum_till_now_right
            end_index = j
    return max_till_now_left + max_till_now_right, start_index, end_index


# ==============================================================

# The iterative method to solve first problem
def find_significant_energy_increase_iterative(A):
    """
    Return a tuple (i,j) where A[i:j] is the most
    significant energy increase period.
    time complexity = O(n)
    """
    a = change_in_energy(A)
    n = len(a)
    max_till_now = -sys.maxsize - 1
    sum_till_now = 0
    start = 0
    for i in range(n):
        sum_till_now += a[i]
        if max_till_now < sum_till_now:
            max_till_now = sum_till_now
            start_index = start
            end_index = i
        elif sum_till_now < 0:
            sum_till_now = 0
            start = i
    if start_index == 0 and end_index == 0:
        return 0, 1
    else:
        return start_index, end_index


# ==============================================================

# The Strassens Algorithm to do the matrix multiplication
def square_matrix_multiply_strassens(A, B):

    A = asarray(A)

    B = asarray(B)

    assert A.shape == B.shape

    assert A.shape == A.T.shape

    assert (len(A) & (len(A) - 1)) == 0, "A is not a power of 2"

    """
    Return the product AB of matrix multiplication.
    Assume len(A) is a power of 2
    """

    n = len(A)
    result = numpy.zeros([n, n])
    if n == 1:
        result[0][0] = A[0][0]*B[0][0]
        return result
    else:
        mid = n//2
        # Initialize A & B's sub matrix of length n/2,
        # to divide given matrices into 4 equal sub matrices.
        A11 = numpy.zeros([mid, mid])
        A12 = numpy.zeros([mid, mid])
        A21 = numpy.zeros([mid, mid])
        A22 = numpy.zeros([mid, mid])

        B11 = numpy.zeros([mid, mid])
        B12 = numpy.zeros([mid, mid])
        B21 = numpy.zeros([mid, mid])
        B22 = numpy.zeros([mid, mid])

        # populate A & B sub matrix
        for i in range(0, mid):
            for j in range(0, mid):
                A11[i][j] = A[i][j]
                A12[i][j] = A[i][j+mid]
                A21[i][j] = A[i+mid][j]
                A22[i][j] = A[i+mid][j+mid]

                B11[i][j] = B[i][j]
                B12[i][j] = B[i][j + mid]
                B21[i][j] = B[i + mid][j]
                B22[i][j] = B[i + mid][j + mid]

        # Calculating Products (P1 - P7) based on Strassen's Formula.
        P1 = square_matrix_multiply_strassens(add_matrices(A11, A22),
                                              add_matrices(B11, B22))
        P2 = square_matrix_multiply_strassens(add_matrices(A21, A22), B11)
        P3 = square_matrix_multiply_strassens(A11, sub_matrices(B12, B22))
        P4 = square_matrix_multiply_strassens(A22, sub_matrices(B21, B11))
        P5 = square_matrix_multiply_strassens(add_matrices(A11, A12), B22)
        P6 = square_matrix_multiply_strassens(sub_matrices(A21, A11),
                                              add_matrices(B11, B12))
        P7 = square_matrix_multiply_strassens(sub_matrices(A12, A22),
                                              add_matrices(B21, B22))

        # Calculating final individual elements resultant matrix

        C11 = sub_matrices(add_matrices(add_matrices(P1, P4), P7), P5)
        C12 = add_matrices(P3, P5)
        C21 = add_matrices(P2, P4)
        C22 = sub_matrices(add_matrices(add_matrices(P1, P3), P6), P2)

        for i in range(0, mid):
            for j in range(0, mid):
                result[i][j] = C11[i][j]
                result[i][j + mid] = C12[i][j]
                result[i + mid][j] = C21[i][j]
                result[i + mid][j + mid] = C22[i][j]
        return numpy.asarray(result, dtype=int)


# method to add two matrices
def add_matrices(a, b):
    n = len(a)
    result = numpy.zeros([n, n])
    for i in range(0, n):
        for j in range(0, n):
            result[i][j] = a[i][j]+b[i][j]
    return result


# method to subtract two matrices
def sub_matrices(a, b):
    n = len(a)
    result = numpy.zeros([n, n])
    for i in range(0, n):
        for j in range(0, n):
            result[i][j] = a[i][j] - b[i][j]
    return result


# ==============================================================


# Calculate the power of a matrix in O(k)
def power_of_matrix_navie(A, k):
    """
    Return A^k.
    time complexity = O(k)
    """
    if not validate_matrix_multiplication(A, k):
        return error_msg
    else:
        result = A
        if k == 1:
            return result
        else:
            while k > 1:
                result = square_matrix_multiply_strassens(A, result)
                k = k-1
    return numpy.asarray(result, dtype=int)


# ==============================================================

# Calculate the power of a matrix in O(log k)
def power_of_matrix_divide_and_conquer(A, k):
    """
    Return A^k.
    time complexity = O(log k)
    """
    if not validate_matrix_multiplication(A, k):
        return error_msg
    else:
        memo = dict()
    return divide_and_conquer(A, k, memo)


# This approach is using Divide & Conquer algorithm in addition to using memoization.
# We are storing the result in each iteration and store its value in a dictionary.
# If we encounter stored value next time, we simply retrieve it from the dictionary
# instead of calculating it again.
def divide_and_conquer(A, k, memo):
    if memo.get(k):
        return memo.get(k)
    elif k == 1:
        memo[k] = A
        return A
    elif k == 2:
        result = square_matrix_multiply_strassens(A, A)
        memo[k] = result
        return result
    else:
        P = divide_and_conquer(A, (k//2), memo)
        Q = divide_and_conquer(A, (k+1)//2, memo)
        result = square_matrix_multiply_strassens(P, Q)
        memo[k] = result
        return numpy.asarray(result, dtype=int)


# ==============================================================

# Function to validate if the input to 2nd question is valid
def validate_matrix_multiplication(A, k):
    n = len(A)
    if k > 10 or k <= 0:
        return False
    if n > 32:
        return False
    if len(A) != len(A[0]):
        return False
    else:
        return True


def test():

    assert(find_significant_energy_increase_brute(ENERGY_LEVEL)
           == (7, 11))
    assert(find_significant_energy_increase_recursive(ENERGY_LEVEL) == (7, 11))
    assert(find_significant_energy_increase_iterative(ENERGY_LEVEL) == (7, 11))
    assert (find_significant_energy_increase_brute(ENERGY_LEVEL_2)
            == (0, 1))
    assert (find_significant_energy_increase_recursive(ENERGY_LEVEL_2) == (0, 1))
    assert (find_significant_energy_increase_iterative(ENERGY_LEVEL_2) == (0, 1))

    assert((square_matrix_multiply_strassens([[0, 1], [1, 1]],
                                             [[0, 1], [1, 1]]) ==
                                             asarray([[1, 1], [1, 2]])).all())
    assert((power_of_matrix_navie([[0, 1], [1, 1]], 3) ==
                                    asarray([[1, 2], [2, 3]])).all())
    assert((power_of_matrix_divide_and_conquer([[0, 1], [1, 1]], 3) ==
                                               asarray([[1, 2], [2, 3]])).all())
    assert ((power_of_matrix_divide_and_conquer([[0, 1], [1, 1]], 0) ==
             error_msg))
    assert ((power_of_matrix_divide_and_conquer([[0, 1], [1, 1]], -2)
             == error_msg))


if __name__ == '__main__':

    test()

# ==============================================================

# print("Test Case ENERGY_LEVEL =", ENERGY_LEVEL)
# print("Most Significant Energy Change using Brute Force")
# print(find_significant_energy_increase_brute(ENERGY_LEVEL))
# print("Most Significant Energy Change using Iteration Method")
# print(find_significant_energy_increase_iterative(ENERGY_LEVEL))
# print("Most Significant Energy Change using recursive Method")
# print(find_significant_energy_increase_recursive(ENERGY_LEVEL))
#

# print("Input Matrix is", a, "and the value of k is", -2)
# print(square_matrix_multiply_strassens(a, a))
# print("Matrix Multiplication using Strassen's formulas implementing Naive Algorithm")
# print(power_of_matrix_navie(a, -2))
# print("Matrix Multiplication using Strassen's formulas implementing Divide and Conquer Algorithm")
#
# print(power_of_matrix_divide_and_conquer(a, -2))
