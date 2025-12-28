
import sympy as sp
from sympy.printing.c import C99CodePrinter
from collections import Counter


class CudaPrinter(C99CodePrinter):
    """Custom printer to optimize math operations for CUDA/C++."""

    def _print_Pow(self, expr):
        if expr.exp == 2:
            base = self._print(expr.base)
            return f"({base} * {base})"
        if expr.exp == 3:
            base = self._print(expr.base)
            return f"({base} * {base} * {base})"
        return super()._print_Pow(expr)


def evaluate_ops_cost(expr_list):
    """Tallies and prints the floating point operation count."""
    total_counts = Counter()
    if not isinstance(expr_list, (list, tuple)):
        expr_list = [expr_list]

    for expr in expr_list:
        ops_dict = sp.count_ops(expr, visual=True)
        if ops_dict.is_Number:
            continue

        args = ops_dict.args if ops_dict.is_Add else [ops_dict]
        for arg in args:
            if arg.is_Mul:
                count, op_type = arg.as_coeff_Mul()
                total_counts[str(op_type)] += int(count)
            else:
                total_counts[str(arg)] += 1

    print("\n" + "=" * 30)
    print(f"{'OPERATION':<15} | {'COUNT':<10}")
    print("-" * 30)
    mapping = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/', 'POW': 'pow', 'log': 'log'}
    grand_total = 0
    for op, count in sorted(total_counts.items()):
        name = mapping.get(op, op)
        print(f"{name:<15} | {count:<10}")
        grand_total += count
    print("-" * 30)
    print(f"{'TOTAL FLOPs':<15} | {grand_total:<10}")
    print("=" * 30 + "\n")


def gipc_indexer(name, i, j):
    """Custom indexer for the matrix class."""
    return f"{name}.m[{i}][{j}]"


def gen_matrix_code(in_var, out_var, in_mat, out_mat, precompute, indexer):
    """Generates optimized C++ code for matrix transformations."""
    printer = CudaPrinter()

    # Map precomputed expressions to temporary symbols
    precompute_sub_map = {expr: sp.symbols(f"precomp_{name}") for expr, name in precompute.items()}
    opt_out_mat = out_mat.subs(precompute_sub_map)

    out_str = ""
    rows, cols = opt_out_mat.shape

    # 1. Input Expansion: Map input matrix to local variables
    for i in range(rows):
        for j in range(cols):
            sym = in_mat[i, j]
            out_str += f"const double {sym.name} = {indexer(in_var, i, j)};\n"
    out_str += "\n"

    # 2. CSE: Common Subexpression Elimination
    flat = [opt_out_mat[i, j] for i in range(rows) for j in range(cols)]
    subs, reduced = sp.cse(flat)

    # print cost analysis
    all_math = [s[1] for s in subs] + reduced
    evaluate_ops_cost(all_math)

    # 3. Intermediate Math: Print temporary CSE variables
    for s in subs:
        var_name = s[0]
        expr_code = printer.doprint(s[1])
        out_str += f"const double {var_name} = {expr_code};\n"
    out_str += "\n"

    # 4. Output Assignment: Map results back to output matrix
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            index_code = indexer(out_var, i, j)
            # Accessing the flattened reduced list directly
            val_code = printer.doprint(reduced[cnt])
            out_str += f"{index_code} = {val_code};\n"
            cnt += 1

    # Final Cleanup: Swap placeholder names for C++ names
    for _, name in precompute.items():
        out_str = out_str.replace(f"precomp_{name}", name)

    print(out_str)


if __name__ == "__main__":
    F = sp.Matrix(3, 3, lambda i, j: sp.symbols(f"F{i}{j}"))
    mu, lmd = sp.symbols("mu lmd")

    det_F = F.det()
    I1 = (F.T * F).trace()
    log_det_F = sp.log(det_F)

    # Energy Density
    E = mu / 2 * (I1 - 3) - mu * log_det_F + lmd / 2 * (log_det_F**2)
    # Stress P = dE/dF
    P = sp.Matrix(3, 3, lambda i, j: sp.diff(E, F[i, j]))

    vecF = [
        F[0, 0],
        F[1, 0],
        F[2, 0],
        F[0, 1],
        F[1, 1],
        F[2, 1],
        F[0, 2],
        F[1, 2],
        F[2, 2],
    ]
    H = sp.hessian(E, vecF)

gen_matrix_code("F", "H", F, H, {}, gipc_indexer)
