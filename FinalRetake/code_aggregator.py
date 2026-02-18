import os
from pathlib import Path

folders = ['navigation', 'scripts', 'tests']

def aggregate_python_files(folders, output_file='aggregated_code.txt'):
    """
    Aggregate all Python files from specified folders into a single text file.
    
    Args:
        folders: List of folder names to search
        output_file: Output file name for aggregated code
    """
    with open(output_file, 'w', encoding='utf-8') as outf:
        for folder in folders:
            if not os.path.isdir(folder):
                print(f"Warning: Folder '{folder}' not found")
                continue
            
            # Walk through the folder and all subfolders
            for root, dirs, files in os.walk(folder):
                # Sort for consistent ordering
                files.sort()
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        
                        # Write path header
                        outf.write(f"\n{'='*80}\n")
                        outf.write(f"FILE: {file_path}\n")
                        outf.write(f"{'='*80}\n\n")
                        
                        # Write file content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                outf.write(content)
                        except Exception as e:
                            outf.write(f"ERROR reading file: {e}\n")
                        
                        outf.write("\n")
    
    print(f"Aggregation complete! Output written to '{output_file}'")

if __name__ == '__main__':
    aggregate_python_files(folders)

