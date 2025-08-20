import os

def write_tree(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = '    ' * level
            f.write(f'{indent}{os.path.basename(dirpath)}/\n')
            subindent = '    ' * (level + 1)
            for filename in filenames:
                f.write(f'{subindent}{filename}\n')

if __name__ == "__main__":
    # Change this to your project root if needed
    root = os.path.dirname(os.path.abspath(__file__))
    write_tree(root, os.path.join(root, 'project_tree.txt'))
    print('Project directory structure written to project_tree.txt')
