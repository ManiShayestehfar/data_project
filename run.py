"""Execute the notebook as the single experiment pipeline."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-process', action='store_true', help='Execute cells through IPython without kernel sockets')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    notebook = nbformat.read(root/'DATA5441_connectivity.ipynb', as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            cell.source = cell.source.replace("MODE = 'quick'", f"MODE = '{args.mode}'")
    try:
        if args.in_process:
            import os
            from IPython.terminal.interactiveshell import TerminalInteractiveShell
            from IPython.utils.capture import capture_output
            os.chdir(root)
            shell = TerminalInteractiveShell.instance()
            shell.run_line_magic('matplotlib', 'inline')
            shell.display_formatter.active_types = ['text/plain', 'text/html', 'image/png', 'image/svg+xml']
            count = 0
            for cell in notebook.cells:
                if cell.cell_type != 'code':
                    continue
                count += 1
                print(f'Executing code cell {count}', flush=True)
                with capture_output() as captured:
                    result = shell.run_cell(cell.source, store_history=True)
                cell.execution_count = count
                if captured.stdout:
                    cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
                if captured.stderr:
                    cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
                for output in captured.outputs:
                    cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
                error = result.error_before_exec or result.error_in_exec
                if error:
                    raise error
        else:
            client = NotebookClient(notebook, timeout=3600, kernel_name='python3', resources={'metadata': {'path': str(root)}})
            client.execute()
    finally:
        nbformat.write(notebook, root/f'DATA5441_connectivity_{args.mode}_executed.ipynb')
    print(f'Completed {args.mode}; results/{args.mode}/')
