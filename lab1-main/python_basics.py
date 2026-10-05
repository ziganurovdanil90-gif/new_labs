"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    count = 0
    for ch in text:
        if ch in "aeiou":
            count += 1
    return count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    seen = []
    for ch in text:
        if ch in seen:
            return False
        seen.append(ch)
    return True


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number > 0:
        if number % 2 == 1:
            count += 1
        number = number // 2
    return count


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        while number > 0:
            product = product * (number % 10)
            number = number // 10
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    total = 0
    for i in range(len(predicted)):
        diff = predicted[i] - expected[i]
        total += diff * diff
    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    result = ""
    divisor = 2
    while number > 1:
        power = 0
        while number % divisor == 0:
            number = number // divisor
            power += 1
        if power == 1:
            result += "(" + str(divisor) + ")"
        elif power > 1:
            result += "(" + str(divisor) + "**" + str(power) + ")"
        divisor += 1
    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)
    n = len(digits)
    side = (n - 1) // 2

    left = 0
    for i in range(side):
        left += int(digits[i])

    right = 0
    for i in range(n - side, n):
        right += int(digits[i])

    return left == right
