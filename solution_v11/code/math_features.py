import ast
import re
from fractions import Fraction

import numpy as np

from common import parse_answer

EXPRESSION = re.compile(r"[-+]?[\d(][\d\s+\-*/().]*")
FRACTION = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def normalize_expression(text):
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    text = re.sub(r"\\(?:times|cdot)\b", "*", text)
    text = re.sub(r"\\div\b", "/", text)
    for _ in range(4):
        text = FRACTION.sub(r"((\1)/(\2))", text)
    text = re.sub(r"\\(?:text|textbf|mathrm|mbox)\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = text.replace("\\", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return text


def evaluate(node):
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = evaluate(node.left)
        right = evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError("Unsupported expression")


def expression_value(segment):
    if re.search(r"\d\s*[a-zA-Z]\b", segment):
        return None
    matches = EXPRESSION.findall(segment)
    if not matches:
        return None
    expression = max(matches, key=len).strip().rstrip("+-*/( ")
    if not expression or not re.search(r"\d", expression):
        return None
    try:
        return evaluate(ast.parse(expression, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


def candidate_equations(text):
    normalized = normalize_expression(text)
    equations = []
    for line in normalized.splitlines():
        if "=" not in line:
            continue
        segments = line.split("=")
        values = [expression_value(segment) for segment in segments]
        for index in range(len(values) - 1):
            left, right = values[index], values[index + 1]
            sources = EXPRESSION.findall(segments[index])
            source = max(sources, key=len) if sources else ""
            has_operator = re.search(r"[+*/]|(?<=\d)\s*-\s*(?=\d)", source)
            if left is None or right is None or not has_operator:
                continue
            tolerance = max(Fraction(1, 1000), max(abs(left), abs(right)) / 1000)
            equations.append((left, right, abs(left - right) <= tolerance, line))
    return equations


def equation_features(bank):
    from independent_features import numbers

    rows = []
    for row in bank:
        question_numbers = set(numbers(row["question"]))
        group = []
        for candidate in row["candidates"]:
            text = candidate["text"]
            equations = candidate_equations(text)
            answer_text = parse_answer(text)
            answer = Fraction(answer_text) if answer_text is not None else None
            checked = len(equations)
            valid = sum(equation[2] for equation in equations)
            invalid = checked - valid
            final_valid = 0
            final_contradicted = 0
            final_matches = 0
            for left, right, is_valid, _ in equations:
                if answer is not None and (left == answer or right == answer):
                    final_matches += 1
                    final_valid += int(is_valid)
                    final_contradicted += int(not is_valid)
            tail_equations = candidate_equations(text[-600:])
            tail_invalid = sum(not equation[2] for equation in tail_equations)
            equation_numbers = set()
            for _, _, _, line in equations:
                equation_numbers.update(numbers(line))
            last = equations[-1] if equations else None
            last_difference = 0.0
            if last is not None and answer is not None:
                last_difference = float(min(abs(last[0] - answer), abs(last[1] - answer)))
            group.append([
                min(checked, 20) / 20.0,
                valid / max(checked, 1),
                invalid / max(checked, 1),
                float(invalid > 0),
                float(last is not None),
                float(last is not None and last[2]),
                min(final_matches, 5) / 5.0,
                min(final_valid, 5) / 5.0,
                min(final_contradicted, 5) / 5.0,
                float(final_valid > 0),
                float(final_contradicted > 0),
                tail_invalid / max(len(tail_equations), 1),
                len(question_numbers & equation_numbers) / max(len(question_numbers), 1),
                len(question_numbers & equation_numbers) / max(len(equation_numbers), 1),
                np.log1p(last_difference) / 8.0,
            ])
        rows.append(group)
    return np.asarray(rows, dtype=np.float32)
