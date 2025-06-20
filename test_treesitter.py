#!/usr/bin/env python3
"""Simple test to verify tree-sitter JavaScript parsing works."""

import tree_sitter
import tree_sitter_javascript

# Test code
js_code = '''
export default function chunkText(text, maxChunkLength = 4000) {
    if (!text || text.length === 0) {
        return [];
    }
    
    const chunks = [];
    let currentIndex = 0;
    
    while (currentIndex < text.length) {
        chunks.push(text.slice(currentIndex, currentIndex + maxChunkLength));
        currentIndex += maxChunkLength;
    }
    
    return chunks;
}

function processData(data) {
    const result = chunkText(data);
    return result;
}

const helper = (input) => {
    return input.trim();
}
'''

def test_treesitter():
    # Initialize tree-sitter
    js_language = tree_sitter.Language(tree_sitter_javascript.language())
    parser = tree_sitter.Parser(js_language)
    
    # Parse the code
    tree = parser.parse(bytes(js_code, "utf8"))
    root_node = tree.root_node
    
    print(f"Root node type: {root_node.type}")
    print(f"Number of children: {len(root_node.children)}")
    
    # Find all function declarations
    def find_functions(node, depth=0):
        indent = "  " * depth
        print(f"{indent}{node.type} ({node.start_point[0]}:{node.start_point[1]} - {node.end_point[0]}:{node.end_point[1]})")
        
        if node.type in ['function_declaration', 'arrow_function', 'function_expression']:
            print(f"{indent}  -> FOUND FUNCTION!")
            # Try to get function name
            for child in node.children:
                if child.type == 'identifier':
                    name = js_code.encode('utf8')[child.start_byte:child.end_byte].decode('utf8')
                    print(f"{indent}     Name: {name}")
                    break
        
        # Handle export statements  
        if node.type == 'export_statement':
            print(f"{indent}  -> FOUND EXPORT!")
            
        # Handle variable declarations with arrow functions
        if node.type == 'lexical_declaration':
            print(f"{indent}  -> FOUND LEXICAL DECLARATION!")
            for child in node.children:
                if child.type == 'variable_declarator':
                    for grandchild in child.children:
                        if grandchild.type == 'arrow_function':
                            print(f"{indent}     Contains arrow function!")
        
        for child in node.children:
            find_functions(child, depth + 1)
    
    find_functions(root_node)

if __name__ == "__main__":
    test_treesitter() 