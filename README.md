# Inverse Function Calculator & Graph Visualizer

Streamlit app that calculates inverse expressions, displays algebraic steps,
plots the original and inverse expressions, and performs symbolic composition
checks.

## Supported input syntax

Use `x` as the variable, `*` for multiplication, and `**` for powers.

Examples:
- `2*x + 4`
- `3*x - 5`
- `x**3`
- `1/x`

This version supports arithmetic expressions using numbers, parentheses, `+`,
`-`, `*`, `/`, and `**`. It does not support function calls such as `sqrt(x)`
in the input field yet.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload `app.py` and `requirements.txt` to GitHub. In Streamlit Community Cloud,
select the repository and use `app.py` as the main file.

## Limitations

This is an educational prototype. A function may not be one-to-one over its
full domain, and some functions have multiple inverse branches or no explicit
inverse supported by SymPy. Domain restrictions may be required.
