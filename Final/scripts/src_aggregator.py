"""
Aggregates all Python source files from kiu_drone_show into a single text file.
"""

from pathlib import Path


def aggregate_src():
    # Get the project root (parent of scripts folder)
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    src_dir = project_root / "kiu_drone_show"
    
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist")
        return
    
    # Collect all Python files
    py_files = sorted(src_dir.glob("*.py"))
    
    if not py_files:
        print(f"No Python files found in {src_dir}")
        return
    
    # Aggregate content
    output = []
    for py_file in py_files:
        output.append(f"{'='*80}\n")
        output.append(f"FILE: {py_file.name}\n")
        output.append(f"{'='*80}\n\n")
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            output.append(content)
        except Exception as e:
            output.append(f"Error reading file: {e}\n")
        
        output.append("\n\n")
    
    # Write to output file
    output_file = project_root / "src_aggregated.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output)
    
    print(f"Successfully aggregated {len(py_files)} files to {output_file}")
    print(f"Total lines: {sum(len(output[i].splitlines()) for i in range(len(output)))}")


if __name__ == "__main__":
    aggregate_src()
