import ast
import operator
import re
import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Inverse Function Calculator",
    page_icon="📈",
    layout="wide"
)

x, y = sp.symbols("x y", real=True)

# Safe arithmetic parser: supports x, numbers, + - * / ** and parentheses.
# This avoids passing arbitrary user input directly to sympify.
_ALLOWED_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def normalize_math_input(text):
    """Convert common textbook notation into safe parser-friendly syntax."""
    text = text.strip().replace("−", "-").replace("×", "*").replace("÷", "/")
    # Convert Unicode superscript powers, e.g. x³ and x², into x**3 and x**2.
    superscript_digits = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
                          "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
    text = re.sub(
        r"([a-zA-Z)])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda match: match.group(1) + "**" + "".join(superscript_digits[c] for c in match.group(2)),
        text,
    )
    # Convert caret notation (x^3) to exponentiation.
    text = text.replace("^", "**")
    # Add multiplication signs for common implicit multiplication: 2x, 3(x+1), x(x+1), (x+1)(x-1).
    text = re.sub(r"(\d)(\s*)(x\b)", r"\1*\3", text)
    text = re.sub(r"(\d|x|\))(\s*)(\()", r"\1*\3", text)
    text = re.sub(r"(\))(\s*)(x\b)", r"\1*\3", text)
    return text


def safe_parse_function(text):
    """Parse a limited, safe arithmetic expression into a SymPy expression."""
    text = normalize_math_input(text)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax.") from exc

    def convert(node):
        if isinstance(node, ast.Expression):
            return convert(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return sp.sympify(node.value)
            raise ValueError("Only numeric constants and x are supported.")

        if isinstance(node, ast.Name):
            if node.id == "x":
                return x
            raise ValueError("Use only the variable x.")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BINARY_OPS:
                raise ValueError("Unsupported operator.")
            left = convert(node.left)
            right = convert(node.right)
            if op_type is ast.Pow:
                if right.is_number and abs(float(right)) > 20:
                    raise ValueError("Exponent is too large for this app.")
            return _ALLOWED_BINARY_OPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY_OPS:
                raise ValueError("Unsupported unary operator.")
            return _ALLOWED_UNARY_OPS[op_type](convert(node.operand))

        raise ValueError(
            "Unsupported expression. Use x, numbers, +, -, multiplication, division, powers and parentheses."
        )

    expression = sp.simplify(convert(tree))
    if expression.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Expression contains an undefined value.")
    return expression


def calculate_inverse(expression):
    """Find inverse candidates and composition checks."""
    # Solve f(y) = x; this directly expresses the inverse in terms of x.
    try:
        solutions = sp.solve(sp.Eq(expression.subs(x, y), x), y)
    except Exception as exc:
        raise ValueError("SymPy could not solve this equation.") from exc

    # Keep only solutions that are explicit expressions in x.
    valid_solutions = []
    for solution in solutions:
        if solution.has(y):
            continue
        if solution.has(sp.ConditionSet):
            continue
        valid_solutions.append(sp.simplify(solution))

    # Remove duplicates.
    unique_solutions = []
    for solution in valid_solutions:
        if not any(sp.simplify(solution - existing) == 0 for existing in unique_solutions):
            unique_solutions.append(solution)

    if len(unique_solutions) != 1:
        if len(unique_solutions) > 1:
            raise ValueError(
                "This equation has multiple inverse branches. "
                "Restrict the original function's domain first."
            )
        raise ValueError(
            "No explicit inverse expression was found for this function. "
            "Try a one-to-one function such as 2x + 4 or x^3."
        )

    inverse = unique_solutions[0]
    first_check = sp.simplify(expression.subs(x, inverse) - x)
    second_check = sp.simplify(inverse.subs(x, expression) - x)
    return inverse, first_check, second_check


def evaluate_function(expression, values):
    """Evaluate a symbolic expression numerically for plotting."""
    numeric_function = sp.lambdify(x, expression, modules="numpy")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = numeric_function(values)

    result = np.asarray(result, dtype=float)
    if result.ndim == 0:
        result = np.full_like(values, result.item())

    result = np.broadcast_to(result, values.shape).copy()
    result[~np.isfinite(result) | (np.abs(result) > 1e5)] = np.nan
    return result


st.title("📈 Inverse Function Calculator")
st.write(
    "Calculate inverse functions, view the algebraic steps, "
    "and compare the function and inverse graphs."
)

with st.sidebar:
    st.header("About")
    st.write("An educational project built with Python, Streamlit, SymPy, NumPy and Matplotlib.")
    st.subheader("Try these examples")
    st.latex(r"2x + 4")
    st.latex(r"3x - 5")
    st.latex(r"x^3")
    st.latex(r"\frac{1}{x}")
    st.caption("Enter expressions in normal math notation, such as 2x + 4, x^3, or 1/x.")

st.subheader("Enter a function")
function_text = st.text_input(
    "Enter a mathematical expression for f(x)",
    value="2x + 4",
    help="Use normal notation: 2x + 4, 3x - 5, x^3, x² + 3x + 2, or 1/x. You do not need to type Python syntax."
)
graph_range = st.slider("Graph range", 5, 30, 10)

if st.button("Calculate Inverse", type="primary", use_container_width=True):
    try:
        expression = safe_parse_function(function_text)
        inverse, first_check, second_check = calculate_inverse(expression)

        st.success("Inverse expression calculated.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Function")
            st.latex(rf"f(x) = {sp.latex(expression)}")
        with col2:
            st.subheader("Inverse Function")
            st.latex(rf"f^{{-1}}(x) = {sp.latex(inverse)}")

        st.divider()
        st.subheader("Step-by-Step Solution")
        st.markdown("**Step 1: Write the original function.**")
        st.latex(rf"y = {sp.latex(expression)}")

        st.markdown("**Step 2: Interchange x and y.**")
        st.latex(rf"x = {sp.latex(expression.subs(x, y))}")

        st.markdown("**Step 3: Solve for y.**")
        st.latex(rf"y = {sp.latex(inverse)}")

        st.markdown("**Step 4: State the inverse.**")
        st.latex(rf"f^{{-1}}(x) = {sp.latex(inverse)}")

        st.divider()
        st.subheader("Graph Visualization")
        values = np.linspace(-graph_range, graph_range, 1000)
        original_values = evaluate_function(expression, values)
        inverse_values = evaluate_function(inverse, values)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(values, original_values, label="Original f(x)", linewidth=2)
        ax.plot(values, inverse_values, label="Inverse f⁻¹(x)", linewidth=2)
        ax.plot(values, values, label="Reference line y = x", linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Original Function and Its Inverse")
        ax.grid(True)
        ax.legend()
        ax.set_xlim(-graph_range, graph_range)
        ax.set_ylim(-graph_range, graph_range)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("Inverse Verification")
        st.markdown("**Check 1: f(f⁻¹(x)) = x**")
        st.latex(rf"f(f^{{-1}}(x)) - x = {sp.latex(first_check)}")
        st.markdown("**Check 2: f⁻¹(f(x)) = x**")
        st.latex(rf"f^{{-1}}(f(x)) - x = {sp.latex(second_check)}")

        if first_check == 0 and second_check == 0:
            st.success("Both inverse identities are satisfied symbolically.")
        else:
            st.warning(
                "The identities could not both be simplified to zero. "
                "The inverse may require domain restrictions."
            )

        st.caption(
            "Important: x**2 is not one-to-one over all real numbers. "
            "Restrict its domain (for example, x >= 0) before treating "
            "sqrt(x) as its inverse."
        )

    except (ValueError, TypeError, sp.SympifyError, NotImplementedError) as exc:
        st.error(f"Could not calculate the inverse: {exc}")
    except Exception as exc:
        # Show a useful diagnostic rather than hiding all errors.
        st.error(f"Unexpected error ({type(exc).__name__}): {exc}")

st.divider()
st.caption("Inverse Function Calculator & Graph Visualizer")
