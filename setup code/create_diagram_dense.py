import graphviz
import os

# --- Styling for Compact Diagrams ---
# Key changes: Larger fonts, smaller node separation
styles = {
    'font_size': '12', # Larger default font size for all nodes
    'input': {'shape': 'oval', 'style': 'filled', 'fillcolor': '#a7c7e7'},
    'output': {'shape': 'oval', 'style': 'filled', 'fillcolor': '#f2a2e8'},
    'conv': {'shape': 'box', 'style': 'filled', 'fillcolor': '#f5d29d'},
    'special_block': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#b3e6b3'},
    'op': {'shape': 'circle', 'fixedsize': 'true', 'width': '0.6', 'fillcolor': '#e6e6e6', 'style': 'filled'},
    'op_cat': {'shape': 'Mdiamond', 'style': 'filled', 'fillcolor': '#d3d3d3'},
}

# --- Graph Attributes for Compact Layout ---
# Key changes: ratio, ranksep, nodesep
compact_graph_attr = {
    'rankdir': 'TB',
    'splines': 'ortho',
    'ratio': 'compress', # Tries to make the drawing as compact as possible
    'ranksep': '0.3',    # Vertical distance between ranks (rows)
    'nodesep': '0.3'     # Horizontal distance between nodes
}

def save_diagram(dot, output_path):
    """Helper function to save the diagram to the specified path."""
    directory, filename = os.path.split(output_path)
    base_name, _ = os.path.splitext(filename)
    # The render command now includes the directory and DPI
    dot.render(os.path.join(directory, base_name), format='png', view=False, cleanup=True)
    print(f"  -> Generated {output_path}")

# ==============================================================================
# --- MODEL 1: VANILLA SRGAN (COMPACT) ---
# ==============================================================================

def generate_srgan_generator_compact(output_path):
    """Generates a compact generator diagram for SRGAN."""
    dot = graphviz.Digraph('SRGAN_Generator', graph_attr={**compact_graph_attr, 'label': 'Model 1: Generator', 'fontsize': '18'})
    for key, style in styles.items():
        if isinstance(style, dict): style['fontsize'] = styles['font_size']

    dot.node('lr_input', 'Low-Res Image', **styles['input'])
    dot.node('init_conv', 'Initial Conv', **styles['conv'])
    dot.node('res_blocks', '16 x Residual Blocks', **styles['special_block'])
    dot.node('post_res_conv', 'Post-Res Conv', **styles['conv'])
    dot.node('add_global', '+', **styles['op'])
    dot.node('upsampling', '2 x Upsampling', **styles['special_block'])
    dot.node('final_conv', 'Final Conv', **styles['conv'])
    dot.node('sr_output', 'Super-Resolved Image', **styles['output'])
    
    dot.edge('lr_input', 'init_conv'); dot.edge('init_conv', 'res_blocks'); dot.edge('init_conv', 'add_global', style='dashed'); dot.edge('res_blocks', 'post_res_conv'); dot.edge('post_res_conv', 'add_global'); dot.edge('add_global', 'upsampling'); dot.edge('upsampling', 'final_conv'); dot.edge('final_conv', 'sr_output')
    save_diagram(dot, output_path)

def generate_srgan_block_detail_compact(output_path):
    """Generates a compact Residual Block detail diagram."""
    dot = graphviz.Digraph('SRGAN_Block_Detail', graph_attr={**compact_graph_attr, 'label': 'Residual Block Detail', 'fontsize': '18'})
    for key, style in styles.items():
        if isinstance(style, dict): style['fontsize'] = styles['font_size']

    dot.node('rb_input', 'Input', **styles['input'])
    dot.node('rb_proc', 'Conv-BN-ReLU\nConv-BN', **styles['conv'])
    dot.node('rb_add', '+', **styles['op'])
    dot.node('rb_output', 'Output', **styles['output'])
    dot.edge('rb_input', 'rb_proc'); dot.edge('rb_proc', 'rb_add'); dot.edge('rb_input', 'rb_add', style='dashed', label='Skip')
    save_diagram(dot, output_path)

def generate_srgan_discriminator_compact(output_path):
    """Generates a compact discriminator diagram for SRGAN."""
    dot = graphviz.Digraph('SRGAN_Discriminator', graph_attr={**compact_graph_attr, 'label': 'Model 1: Discriminator', 'fontsize': '18'})
    for key, style in styles.items():
        if isinstance(style, dict): style['fontsize'] = styles['font_size']

    dot.node('d_input', 'Image', **styles['input'])
    dot.node('d_convs', 'Conv Blocks', **styles['conv'])
    dot.node('d_dense', 'Dense Layers', **styles['conv'])
    dot.node('d_output', 'Score', **styles['output'])
    dot.edge('d_input', 'd_convs'); dot.edge('d_convs', 'd_dense'); dot.edge('d_dense', 'd_output')
    save_diagram(dot, output_path)

# ==============================================================================
# --- MODEL 2: ESRGAN-STYLE (COMPACT) ---
# ==============================================================================

def generate_esrgan_generator_compact(output_path):
    """Generates a compact generator diagram for ESRGAN."""
    dot = graphviz.Digraph('ESRGAN_Generator', graph_attr={**compact_graph_attr, 'label': 'Model 2: Generator', 'fontsize': '18'})
    for key, style in styles.items():
        if isinstance(style, dict): style['fontsize'] = styles['font_size']

    dot.node('lr_input', 'Low-Res Image', **styles['input'])
    dot.node('init_conv', 'Initial Conv', **styles['conv'])
    dot.node('rrdb_blocks', '23 x RRDBs', **styles['special_block'])
    dot.node('post_rrdb_conv', 'Post-RRDB Conv', **styles['conv'])
    dot.node('add_global', '+', **styles['op'])
    dot.node('upsampling', '2 x Upsampling', **styles['special_block'])
    dot.node('final_convs', 'Refinement', **styles['conv'])
    dot.node('sr_output', 'Super-Resolved\nImage', **styles['output'])
    
    dot.edge('lr_input', 'init_conv'); dot.edge('init_conv', 'rrdb_blocks'); dot.edge('init_conv', 'add_global', style='dashed'); dot.edge('rrdb_blocks', 'post_rrdb_conv'); dot.edge('post_rrdb_conv', 'add_global'); dot.edge('add_global', 'upsampling'); dot.edge('upsampling', 'final_convs'); dot.edge('final_convs', 'sr_output')
    save_diagram(dot, output_path)

def generate_esrgan_block_detail_compact(output_path):
    """Generates a compact RRDB detail diagram."""
    dot = graphviz.Digraph('ESRGAN_Block_Detail', graph_attr={**compact_graph_attr, 'label': 'RRDB Detail', 'fontsize': '18'})
    for key, style in styles.items():
        if isinstance(style, dict): style['fontsize'] = styles['font_size']

    dot.node('rrdb_input', 'Input', **styles['input'])
    dot.node('drb_stack', '3 x Dense Blocks', **styles['special_block'])
    dot.node('rrdb_add', '+\n(beta=0.2)', **styles['op'])
    dot.node('rrdb_output', 'Output', **styles['output'])
    dot.edge('rrdb_input', 'drb_stack'); dot.edge('drb_stack', 'rrdb_add'); dot.edge('rrdb_input', 'rrdb_add', style='dashed', label='Skip')
    save_diagram(dot, output_path)

def generate_dense_block_detail_compact(output_path):
    """Generates a compact Dense Block detail diagram."""
    dot = graphviz.Digraph('ESRGAN_Dense_Block', graph_attr={**compact_graph_attr, 'label': 'Dense Block Detail', 'fontsize': '18'})
    for key, style in styles.items():
        if isinstance(style, dict): style['fontsize'] = styles['font_size']
        
    dot.node('db_input', 'Input x₀', **styles['input'])
    dot.node('conv1', 'Conv 1', **styles['conv'])
    dot.node('cat2', 'Concat', **styles['op_cat'])
    dot.node('conv2', 'Conv 2', **styles['conv'])
    dot.node('cat_final', '...\nConcat', **styles['op_cat'])
    dot.node('conv_final', 'Final Conv', **styles['conv'])
    dot.node('db_add', '+', **styles['op'])
    dot.node('db_output', 'Output', **styles['output'])

    dot.edge('db_input', 'conv1')
    dot.edge('db_input', 'cat2', style='dashed')
    dot.edge('conv1', 'cat2')
    dot.edge('cat2', 'conv2')
    dot.edge('conv2', 'cat_final')
    dot.edge('cat_final', 'conv_final')
    dot.edge('conv_final', 'db_add')
    dot.edge('db_input', 'db_add', style='dashed', label='Skip')
    dot.edge('db_add', 'db_output')
    save_diagram(dot, output_path)

def generate_esrgan_discriminator_compact(output_path):
    """Generates a compact RaGAN discriminator diagram."""
    dot = graphviz.Digraph('ESRGAN_Discriminator', graph_attr={**compact_graph_attr, 'label': 'Model 2: Discriminator', 'fontsize': '18'})
    for key, style in styles.items():
        if isinstance(style, dict): style['fontsize'] = styles['font_size']

    dot.node('d_input', 'Image', **styles['input'])
    dot.node('d_convs', 'Conv Blocks', **styles['conv'])
    dot.node('d_dense', 'Dense Layers', **styles['conv'])
    dot.node('d_output', 'Logit', **styles['output'])
    dot.edge('d_input', 'd_convs'); dot.edge('d_convs', 'd_dense'); dot.edge('d_dense', 'd_output')
    save_diagram(dot, output_path)

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

def main():
    """Main function to create directories and generate all diagrams."""
    model1_dir = 'model_1_compact_diagrams'
    model2_dir = 'model_2_compact_diagrams'
    
    os.makedirs(model1_dir, exist_ok=True)
    os.makedirs(model2_dir, exist_ok=True)
    
    print(f"--- Generating COMPACT diagrams in ./{model1_dir} ---")
    generate_srgan_generator_compact(os.path.join(model1_dir, 'generator.png'))
    generate_srgan_block_detail_compact(os.path.join(model1_dir, 'block_detail_residual.png'))
    generate_srgan_discriminator_compact(os.path.join(model1_dir, 'discriminator.png'))
    
    print(f"\n--- Generating COMPACT diagrams in ./{model2_dir} ---")
    generate_esrgan_generator_compact(os.path.join(model2_dir, 'generator.png'))
    generate_esrgan_block_detail_compact(os.path.join(model2_dir, 'block_detail_rrdb.png'))
    generate_dense_block_detail_compact(os.path.join(model2_dir, 'block_detail_dense.png'))
    generate_esrgan_discriminator_compact(os.path.join(model2_dir, 'discriminator_ragan.png'))
    
    print("\nAll compact diagrams generated successfully.")

if __name__ == '__main__':
    main()