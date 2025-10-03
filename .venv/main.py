from nicegui import ui
from fractions import Fraction


class MatrixOps:
    @staticmethod
    def transpose(m):
        rows, cols = len(m), len(m[0])
        result = [[0] * rows for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                result[j][i] = m[i][j]
        return result

    @staticmethod
    def add(a, b):
        rows, cols = len(a), len(a[0])
        return [[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)]

    @staticmethod
    def subtract(a, b):
        rows, cols = len(a), len(a[0])
        return [[a[i][j] - b[i][j] for j in range(cols)] for i in range(rows)]

    @staticmethod
    def multiply(a, b):
        rows_a, cols_a = len(a), len(a[0])
        cols_b = len(b[0])
        result = [[0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
        return result

    @staticmethod
    def scalar_mult(m, s):
        return [[m[i][j] * s for j in range(len(m[0]))] for i in range(len(m))]

    @staticmethod
    def determinant(m):
        n = len(m)
        if n == 1:
            return m[0][0]
        if n == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        # Copy matrix
        mat = [row[:] for row in m]
        det = 1

        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(mat[k][i]) > abs(mat[max_row][i]):
                    max_row = k

            if abs(mat[max_row][i]) < 1e-10:
                return 0

            if max_row != i:
                mat[i], mat[max_row] = mat[max_row], mat[i]
                det *= -1

            det *= mat[i][i]

            for k in range(i + 1, n):
                factor = mat[k][i] / mat[i][i]
                for j in range(i, n):
                    mat[k][j] -= factor * mat[i][j]

        return det

    @staticmethod
    def inverse(m):
        n = len(m)
        mat = [row[:] for row in m]
        inv = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(mat[k][i]) > abs(mat[max_row][i]):
                    max_row = k

            if abs(mat[max_row][i]) < 1e-10:
                return None

            mat[i], mat[max_row] = mat[max_row], mat[i]
            inv[i], inv[max_row] = inv[max_row], inv[i]

            pivot = mat[i][i]
            for j in range(n):
                mat[i][j] /= pivot
                inv[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = mat[k][i]
                    for j in range(n):
                        mat[k][j] -= factor * mat[i][j]
                        inv[k][j] -= factor * inv[i][j]

        return inv

    @staticmethod
    def trace(m):
        return sum(m[i][i] for i in range(len(m)))

    @staticmethod
    def rank(m):
        mat = [row[:] for row in m]
        rows, cols = len(mat), len(mat[0])
        rank = 0

        for col in range(cols):
            if rank >= rows:
                break

            pivot_row = rank
            for row in range(rank + 1, rows):
                if abs(mat[row][col]) > abs(mat[pivot_row][col]):
                    pivot_row = row

            if abs(mat[pivot_row][col]) < 1e-10:
                continue

            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]

            for row in range(rank + 1, rows):
                factor = mat[row][col] / mat[rank][col]
                for c in range(cols):
                    mat[row][c] -= factor * mat[rank][c]

            rank += 1

        return rank


def format_number(val, use_decimals):
    """Format a number as integer, fraction, or decimal based on preference"""
    if use_decimals:
        # Decimal mode
        #If it's an integer, return the integer, don't force convert to a decimal
        '''if abs(val - round(val)) < 1e-10:
            return str(int(round(val)))
        else:
        ''' # Intended to only make it decimal if it absolutely needs to be
        return f'{val:.4f}'
    else:
        # Integer/Fraction mode
        # Check if it's essentially an integer
        if abs(val - round(val)) < 1e-10:
            return str(int(round(val)))
        else:
            # Try to represent as fraction
            try:
                frac = Fraction(val).limit_denominator(1000)
                # If fraction is very close to original value, use it
                if abs(float(frac) - val) < 1e-10:
                    return str(frac)
                else:
                    # Fall back to decimal if fraction doesn't work well
                    return f'{val:.4f}'
            except:
                return f'{val:.4f}'


def main():
    ops = MatrixOps()
    use_decimals = {'value': False}  # Store state in dict to allow modification in nested functions

    ui.colors(primary='#3b82f6', secondary='#8b5cf6', positive='#10b981')

    with ui.header().classes('bg-blue-600'):
        ui.label('Matrix Calculator').classes('text-2xl font-bold')

    with ui.column().classes('p-6 max-w-6xl mx-auto gap-4'):

        # Display mode toggle
        with ui.card().classes('p-4 bg-gray-50'):
            with ui.row().classes('items-center gap-4'):
                ui.label('Display Mode:').classes('font-bold')
                mode_switch = ui.switch('Decimal Mode', value=False)
                ui.label('').classes('text-sm text-gray-600')

                def toggle_mode(e):
                    use_decimals['value'] = e.value

                mode_switch.on_value_change(toggle_mode)

        # Size controls
        with ui.card().classes('p-4'):
            ui.label('Matrix Dimensions').classes('text-lg font-bold mb-2')
            with ui.row().classes('gap-4'):
                with ui.column():
                    ui.label('Matrix A')
                    rows_a = ui.input('Rows', value='3').classes('w-24').props('dense outlined')
                    cols_a = ui.input('Cols', value='3').classes('w-24').props('dense outlined')
                with ui.column():
                    ui.label('Matrix B')
                    rows_b = ui.input('Rows', value='3').classes('w-24').props('dense outlined')
                    cols_b = ui.input('Cols', value='3').classes('w-24').props('dense outlined')

        # Matrix inputs
        with ui.row().classes('gap-4'):
            with ui.card().classes('p-4'):
                ui.label('Matrix A').classes('text-xl font-bold mb-2')
                matrix_a_container = ui.column()
                matrix_a_inputs = []
                for i in range(5):
                    row = []
                    with matrix_a_container:
                        with ui.row().classes('gap-1') as row_ui:
                            for j in range(5):
                                inp = ui.input(value='0').classes('w-16 text-center').props('dense outlined')
                                row.append(inp)
                            row.append(row_ui)
                    matrix_a_inputs.append(row)

                with ui.row().classes('gap-2 mt-2'):
                    def clear_a():
                        try:
                            r = max(1, min(5, int(rows_a.value or 3)))
                            c = max(1, min(5, int(cols_a.value or 3)))
                            for i in range(r):
                                for j in range(c):
                                    matrix_a_inputs[i][j].value = '0'
                        except:
                            pass

                    def identity_a():
                        try:
                            r = max(1, min(5, int(rows_a.value or 3)))
                            c = max(1, min(5, int(cols_a.value or 3)))
                            for i in range(r):
                                for j in range(c):
                                    matrix_a_inputs[i][j].value = '1' if i == j else '0'
                        except:
                            pass

                    ui.button('Clear', on_click=clear_a).props('size=sm')
                    ui.button('Identity', on_click=identity_a).props('size=sm')

            with ui.card().classes('p-4'):
                ui.label('Matrix B').classes('text-xl font-bold mb-2')
                matrix_b_container = ui.column()
                matrix_b_inputs = []
                for i in range(5):
                    row = []
                    with matrix_b_container:
                        with ui.row().classes('gap-1') as row_ui:
                            for j in range(5):
                                inp = ui.input(value='0').classes('w-16 text-center').props('dense outlined')
                                row.append(inp)
                            row.append(row_ui)
                    matrix_b_inputs.append(row)

                with ui.row().classes('gap-2 mt-2'):
                    def clear_b():
                        try:
                            r = max(1, min(5, int(rows_b.value or 3)))
                            c = max(1, min(5, int(cols_b.value or 3)))
                            for i in range(r):
                                for j in range(c):
                                    matrix_b_inputs[i][j].value = '0'
                        except:
                            pass

                    def identity_b():
                        try:
                            r = max(1, min(5, int(rows_b.value or 3)))
                            c = max(1, min(5, int(cols_b.value or 3)))
                            for i in range(r):
                                for j in range(c):
                                    matrix_b_inputs[i][j].value = '1' if i == j else '0'
                        except:
                            pass

                    ui.button('Clear', on_click=clear_b).props('size=sm')
                    ui.button('Identity', on_click=identity_b).props('size=sm')

        # Update visibility
        def update_visibility_a():
            try:
                r = max(1, min(5, int(rows_a.value or 3)))
                c = max(1, min(5, int(cols_a.value or 3)))
            except:
                r, c = 3, 3
            for i in range(5):
                matrix_a_inputs[i][-1].set_visibility(i < r)
                for j in range(5):
                    matrix_a_inputs[i][j].set_visibility(i < r and j < c)

        def update_visibility_b():
            try:
                r = max(1, min(5, int(rows_b.value or 3)))
                c = max(1, min(5, int(cols_b.value or 3)))
            except:
                r, c = 3, 3
            for i in range(5):
                matrix_b_inputs[i][-1].set_visibility(i < r)
                for j in range(5):
                    matrix_b_inputs[i][j].set_visibility(i < r and j < c)

        rows_a.on_value_change(lambda: update_visibility_a())
        cols_a.on_value_change(lambda: update_visibility_a())
        rows_b.on_value_change(lambda: update_visibility_b())
        cols_b.on_value_change(lambda: update_visibility_b())
        update_visibility_a()
        update_visibility_b()

        # Helper functions
        def get_matrix_a():
            try:
                r = max(1, min(5, int(rows_a.value or 3)))
                c = max(1, min(5, int(cols_a.value or 3)))
            except:
                r, c = 3, 3
            result = []
            for i in range(r):
                row = []
                for j in range(c):
                    try:
                        val = float(matrix_a_inputs[i][j].value or 0)
                    except:
                        val = 0
                    row.append(val)
                result.append(row)
            return result

        def get_matrix_b():
            try:
                r = max(1, min(5, int(rows_b.value or 3)))
                c = max(1, min(5, int(cols_b.value or 3)))
            except:
                r, c = 3, 3
            result = []
            for i in range(r):
                row = []
                for j in range(c):
                    try:
                        val = float(matrix_b_inputs[i][j].value or 0)
                    except:
                        val = 0
                    row.append(val)
                result.append(row)
            return result

        def show_matrix(mat, title):
            result_area.clear()
            with result_area:
                with ui.card().classes('p-4'):
                    ui.label(title).classes('text-lg font-bold mb-2')
                    with ui.grid(columns=len(mat[0])).classes('gap-2'):
                        for row in mat:
                            for val in row:
                                formatted = format_number(val, use_decimals['value'])
                                ui.label(formatted).classes('w-20 text-center font-mono')

        def show_scalar(val, title):
            result_area.clear()
            with result_area:
                with ui.card().classes('p-4'):
                    ui.label(title).classes('text-lg font-bold mb-2')
                    formatted = format_number(val, use_decimals['value'])
                    ui.label(formatted).classes('text-2xl font-mono')

        def show_error(msg):
            result_area.clear()
            with result_area:
                ui.label(f'❌ {msg}').classes('text-red-600 font-bold')

        # Operations
        with ui.card().classes('p-4'):
            ui.label('Operations').classes('text-xl font-bold mb-2')

            ui.label('Single Matrix (A):').classes('font-bold mt-2')
            with ui.row().classes('gap-2 flex-wrap'):
                def do_transpose():
                    show_matrix(ops.transpose(get_matrix_a()), 'Transpose of A')

                def do_determinant():
                    m = get_matrix_a()
                    if len(m) != len(m[0]):
                        show_error('Matrix must be square')
                        return
                    show_scalar(ops.determinant(m), 'Determinant of A')

                def do_inverse():
                    m = get_matrix_a()
                    if len(m) != len(m[0]):
                        show_error('Matrix must be square')
                        return
                    inv = ops.inverse(m)
                    if inv is None:
                        show_error('Matrix is singular (not invertible)')
                    else:
                        show_matrix(inv, 'Inverse of A')

                def do_trace():
                    m = get_matrix_a()
                    if len(m) != len(m[0]):
                        show_error('Matrix must be square')
                        return
                    show_scalar(ops.trace(m), 'Trace of A')

                def do_rank():
                    show_scalar(ops.rank(get_matrix_a()), 'Rank of A')

                ui.button('Transpose', on_click=do_transpose)
                ui.button('Determinant', on_click=do_determinant)
                ui.button('Inverse', on_click=do_inverse)
                ui.button('Trace', on_click=do_trace)
                ui.button('Rank', on_click=do_rank)

            ui.label('Two Matrices:').classes('font-bold mt-4')
            with ui.row().classes('gap-2 flex-wrap'):
                def do_add():
                    a, b = get_matrix_a(), get_matrix_b()
                    if len(a) != len(b) or len(a[0]) != len(b[0]):
                        show_error('Matrices must have same dimensions')
                        return
                    show_matrix(ops.add(a, b), 'A + B')

                def do_subtract():
                    a, b = get_matrix_a(), get_matrix_b()
                    if len(a) != len(b) or len(a[0]) != len(b[0]):
                        show_error('Matrices must have same dimensions')
                        return
                    show_matrix(ops.subtract(a, b), 'A - B')

                def do_multiply():
                    a, b = get_matrix_a(), get_matrix_b()
                    if len(a[0]) != len(b):
                        show_error('Columns of A must equal rows of B')
                        return
                    show_matrix(ops.multiply(a, b), 'A × B')

                ui.button('A + B', on_click=do_add)
                ui.button('A - B', on_click=do_subtract)
                ui.button('A × B', on_click=do_multiply)

            ui.label('Scalar Operations:').classes('font-bold mt-4')
            with ui.row().classes('gap-2'):
                scalar = ui.input('Scalar', value='2').classes('w-24').props('dense outlined')

                def do_scalar():
                    try:
                        scalar_val = float(scalar.value or 1)
                    except:
                        scalar_val = 1
                    show_matrix(ops.scalar_mult(get_matrix_a(), scalar_val),
                                f'{scalar_val} × A')

                ui.button('Multiply A', on_click=do_scalar)

        # Results
        with ui.card().classes('p-4'):
            ui.label('Result').classes('text-xl font-bold mb-2')
            result_area = ui.column()
            with result_area:
                ui.label('Select an operation above').classes('text-gray-500')


if __name__ in {"__main__", "__mp_main__"}:
    main()
    ui.run(title='Matrix Calculator', port=8080)