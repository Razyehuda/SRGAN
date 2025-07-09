import graphviz
import os

# --- Styling ---
# Define a consistent style dictionary for all diagram nodes.
styles = {
    'input': {'shape': 'oval', 'style': 'filled', 'fillcolor': '#a7c7e7'},
    'output': {'shape': 'oval', 'style': 'filled', 'fillcolor': '#f2a2e8'},
    'conv': {'shape': 'box', 'style': 'filled', 'fillcolor': '#f5d29d'},
    'special_block': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#b3e6b3'},
    'op': {'shape': 'circle', 'fixedsize': 'true', 'width': '0.7', 'fillcolor': '#e6e6e6', 'style': 'filled'},
    'op_cat': {'shape': 'Mdiamond', 'style': 'filled', 'fillcolor': '#d3d3d3'}, # Diamond for concatenation
    'label': {'shape': 'plaintext', 'fontsize': '10'}
}

def save_diagram(dot, output_path):
    """Helper function to correctly save the diagram to the specified path."""
    directory, filename = os.path.split(output_path)
    base_name, _ = os.path.splitext(filename)
    dot.render(os.path.join(directory, base_name), format='png', view=False, cleanup=True)
    print(f"  -> Generated {output_path}")

# ==============================================================================
# --- MODEL 1: VANILLA SRGAN ---
# ==============================================================================

def generate_srgan_generator(output_path):
    """Generates the main generator architecture for SRGAN."""
    dot = graphviz.Digraph('SRGAN_Generator', graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.8', 'label': 'Model 1: Generator (SRGAN)', 'fontsize': '16'})
    dot.node('lr_input', 'Low-Res Image', **styles['input'])
    dot.node('init_conv', 'Initial Conv', **styles['conv'])
    dot.node('res_blocks', '16 x Residual Blocks', **styles['special_block'])
    dot.node('post_res_conv', 'Post-Res Conv', **styles['conv'])
    dot.node('add_global', '+', **styles['op'])
    dot.node('upsampling', '2 x Upsampling\n(PixelShuffle)', **styles['special_block'])
    dot.node('final_conv', 'Final Conv', **styles['conv'])
    dot.node('sr_output', 'Super-Resolved Image', **styles['output'])
    dot.edge('lr_input', 'init_conv'); dot.edge('init_conv', 'res_blocks'); dot.edge('init_conv', 'add_global', style='dashed', arrowhead='none'); dot.edge('res_blocks', 'post_res_conv'); dot.edge('post_res_conv', 'add_global'); dot.edge('add_global', 'upsampling'); dot.edge('upsampling', 'final_conv'); dot.edge('final_conv', 'sr_output')
    save_diagram(dot, output_path)

def generate_srgan_block_detail(output_path):
    """Generates the detail diagram for a single Residual Block."""
    dot = graphviz.Digraph('SRGAN_Block_Detail', graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'label': 'Model 1: Residual Block Detail', 'fontsize': '16'})
    dot.node('rb_input', 'Input to Block', **styles['input'])
    dot.node('rb_proc', 'Conv -> BN -> ReLU\n-> Conv -> BN', **styles['conv'])
    dot.node('rb_add', '+', **styles['op'])
    dot.node('rb_output', 'Output of Block', **styles['output'])
    dot.edge('rb_input', 'rb_proc'); dot.edge('rb_proc', 'rb_add'); dot.edge('rb_input', 'rb_add', style='dashed', label='  Skip Connection'); dot.edge('rb_add', 'rb_output')
    save_diagram(dot, output_path)

def generate_srgan_discriminator(output_path):
    """Generates the discriminator architecture for SRGAN."""
    dot = graphviz.Digraph('SRGAN_Discriminator', graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'label': 'Model 1: Discriminator', 'fontsize': '16'})
    dot.node('d_input', 'Image\n(Real or Fake)', **styles['input'])
    dot.node('d_convs', '8 x Conv Blocks', **styles['conv'])
    dot.node('d_dense', 'Dense Layers', **styles['conv'])
    dot.node('d_output', 'Real/Fake Score\n(Sigmoid)', **styles['output'])
    dot.edge('d_input', 'd_convs'); dot.edge('d_convs', 'd_dense'); dot.edge('d_dense', 'd_output')
    save_diagram(dot, output_path)

# ==============================================================================
# --- MODEL 2: ESRGAN-STYLE ---
# ==============================================================================

def generate_esrgan_generator(output_path):
    """Generates the main generator architecture for ESRGAN."""
    dot = graphviz.Digraph('ESRGAN_Generator', graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.8', 'label': 'Model 2: Generator (ESRGAN)', 'fontsize': '16'})
    dot.node('lr_input', 'Low-Res Image', **styles['input'])
    dot.node('init_conv', 'Initial Conv', **styles['conv'])
    dot.node('rrdb_blocks', '23 x RRDBs\n(No Batch Norm)', **styles['special_block'])
    dot.node('post_rrdb_conv', 'Post-RRDB Conv', **styles['conv'])
    dot.node('add_global', '+', **styles['op'])
    dot.node('upsampling', '2 x Upsampling\n(PixelShuffle)', **styles['special_block'])
    dot.node('final_convs', 'Refinement Convs', **styles['conv'])
    dot.node('sr_output', 'Super-Resolved Image', **styles['output'])
    dot.edge('lr_input', 'init_conv'); dot.edge('init_conv', 'rrdb_blocks'); dot.edge('init_conv', 'add_global', style='dashed', arrowhead='none'); dot.edge('rrdb_blocks', 'post_rrdb_conv'); dot.edge('post_rrdb_conv', 'add_global'); dot.edge('add_global', 'upsampling'); dot.edge('upsampling', 'final_convs'); dot.edge('final_convs', 'sr_output')
    save_diagram(dot, output_path)

def generate_esrgan_block_detail(output_path):
    """Generates the detail diagram for a single RRDB."""
    dot = graphviz.Digraph('ESRGAN_Block_Detail', graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'label': 'Model 2: RRDB Detail', 'fontsize': '16'})
    dot.node('rrdb_input', 'Input to RRDB', **styles['input'])
    dot.node('drb_stack', '3 x Dense Blocks\n(See Dense Block diagram for details)', **styles['special_block'])
    dot.node('rrdb_add', '+\n(beta=0.2)', **styles['op'])
    dot.node('rrdb_output', 'Output of RRDB', **styles['output'])
    dot.edge('rrdb_input', 'drb_stack'); dot.edge('drb_stack', 'rrdb_add'); dot.edge('rrdb_input', 'rrdb_add', style='dashed', label='  Skip Connection')
    save_diagram(dot, output_path)

def generate_dense_block_detail(output_path):
    """NEW: Generates the detail diagram for a single Dense Block."""
    dot = graphviz.Digraph('ESRGAN_Dense_Block_Detail', graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.5', 'label': 'Model 2: Dense Block Detail', 'fontsize': '16'})
    
    # Define nodes
    dot.node('db_input', 'Input x₀', **styles['input'])
    dot.node('conv1', 'Conv 1', **styles['conv'])
    dot.node('cat2', 'Concat', **styles['op_cat'])
    dot.node('conv2', 'Conv 2', **styles['conv'])
    dot.node('cat3', 'Concat', **styles['op_cat'])
    dot.node('conv3', 'Conv 3', **styles['conv'])
    dot.node('cat_final', 'Concat', **styles['op_cat'])
    dot.node('conv_final', 'Final Conv', **styles['conv'])
    dot.node('db_add', '+ (beta=0.2)', **styles['op'])
    dot.node('db_output', 'Output', **styles['output'])

    # Connect the flow, showing concatenation at each step
    dot.edge('db_input', 'conv1')
    
    dot.edge('db_input', 'cat2', style='dashed')
    dot.edge('conv1', 'cat2')
    dot.edge('cat2', 'conv2')

    dot.edge('db_input', 'cat3', style='dashed')
    dot.edge('conv1', 'cat3', style='dashed')
    dot.edge('conv2', 'cat3')
    dot.edge('cat3', 'conv3')
    
    # The final concatenation before the last conv layer
    # Simplified to show the concept
    dot.edge('db_input', 'cat_final', style='dashed')
    dot.edge('conv1', 'cat_final', style='dashed')
    dot.edge('conv2', 'cat_final', style='dashed')
    dot.edge('conv3', 'cat_final', label='...etc.', style='dashed')
    dot.edge('cat_final', 'conv_final')
    
    # Final residual connection
    dot.edge('conv_final', 'db_add')
    dot.edge('db_input', 'db_add', style='dashed', label='  Skip Connection')
    dot.edge('db_add', 'db_output')

    save_diagram(dot, output_path)


def generate_esrgan_discriminator(output_path):
    """Generates the discriminator architecture for RaGAN."""
    dot = graphviz.Digraph('ESRGAN_Discriminator', graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'label': 'Model 2: Discriminator (RaGAN)', 'fontsize': '16'})
    dot.node('d_input', 'Image\n(Real or Fake)', **styles['input'])
    dot.node('d_convs', '8 x Conv Blocks', **styles['conv'])
    dot.node('d_dense', 'Dense Layers', **styles['conv'])
    dot.node('d_output', 'Realism Logit\n(No Sigmoid)', **styles['output'])
    dot.edge('d_input', 'd_convs'); dot.edge('d_convs', 'd_dense'); dot.edge('d_dense', 'd_output')
    save_diagram(dot, output_path)

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

def main():
    """Main function to create directories and generate all diagrams."""
    model1_dir = 'model_1_diagrams'
    model2_dir = 'model_2_diagrams'
    
    os.makedirs(model1_dir, exist_ok=True)
    os.makedirs(model2_dir, exist_ok=True)
    
    print("--- Generating diagrams for Model 1 (SRGAN) ---")
    generate_srgan_generator(os.path.join(model1_dir, 'generator.png'))
    generate_srgan_block_detail(os.path.join(model1_dir, 'block_detail_residual.png'))
    generate_srgan_discriminator(os.path.join(model1_dir, 'discriminator.png'))
    
    print("\n--- Generating diagrams for Model 2 (ESRGAN) ---")
    generate_esrgan_generator(os.path.join(model2_dir, 'generator.png'))
    generate_esrgan_block_detail(os.path.join(model2_dir, 'block_detail_rrdb.png'))
    generate_dense_block_detail(os.path.join(model2_dir, 'block_detail_dense.png')) # <-- NEW DIAGRAM
    generate_esrgan_discriminator(os.path.join(model2_dir, 'discriminator_ragan.png'))
    
    print("\nAll diagrams generated successfully.")

if __name__ == '__main__':
    main()