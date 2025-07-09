import graphviz

# --- Styling ---
# Define consistent styles for different node types to make diagrams readable.
styles = {
    'input': {'shape': 'oval', 'style': 'filled', 'fillcolor': '#a7c7e7'},       # Light blue
    'output': {'shape': 'oval', 'style': 'filled', 'fillcolor': '#f2a2e8'},      # Light pink
    'conv': {'shape': 'box', 'style': 'filled', 'fillcolor': '#f5d29d'},          # Light orange
    'special_block': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#b3e6b3'}, # Light green
    'op': {'shape': 'circle', 'fixedsize': 'true', 'width': '0.6', 'fillcolor': '#e6e6e6', 'style': 'filled'},
    'label': {'shape': 'plaintext'}
}

def create_srgan_diagram():
    """Generates the diagram for Model 1: Vanilla SRGAN."""
    dot = graphviz.Digraph(
        'SRGAN_Model',
        comment='Model 1: Vanilla SRGAN Architecture',
        graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.8'}
    )

    # --- Generator Subgraph ---
    with dot.subgraph(name='cluster_generator') as g:
        g.attr(label='Generator (Model 1)', style='filled', color='lightgrey')
        
        # Input
        g.node('lr_input', 'Low-Res Image', **styles['input'])

        # Main Path
        g.node('init_conv', 'Initial Conv', **styles['conv'])
        g.node('res_blocks', '16 x Residual Blocks', **styles['special_block'])
        g.node('post_res_conv', 'Post-Res Conv', **styles['conv'])
        g.node('add1', '+', **styles['op'])
        g.node('upsample1', 'Upsampling (PixelShuffle)', **styles['special_block'])
        g.node('upsample2', 'Upsampling (PixelShuffle)', **styles['special_block'])
        g.node('final_conv', 'Final Conv', **styles['conv'])
        g.node('sr_output', 'Super-Resolved Image', **styles['output'])

        # Residual Block detail (as a separate cluster for clarity)
        with g.subgraph(name='cluster_rb') as rb:
            rb.attr(label='Residual Block Detail', style='filled', color='#f0f0f0')
            rb.node('rb_conv1', 'Conv 3x3', **styles['conv'])
            rb.node('rb_bn1', 'BatchNorm', **styles['conv'])
            rb.node('rb_relu1', 'ReLU', **styles['conv'])
            rb.node('rb_conv2', 'Conv 3x3', **styles['conv'])
            rb.node('rb_bn2', 'BatchNorm', **styles['conv'])
            rb.node('rb_add', '+', **styles['op'])
            rb.edge('res_blocks', 'rb_conv1', style='dashed', label='(example)')
            rb.edge('rb_conv1', 'rb_bn1')
            rb.edge('rb_bn1', 'rb_relu1')
            rb.edge('rb_relu1', 'rb_conv2')
            rb.edge('rb_conv2', 'rb_bn2')
            rb.edge('rb_bn2', 'rb_add')

        # Connect generator parts
        g.edge('lr_input', 'init_conv')
        g.edge('init_conv', 'res_blocks')
        g.edge('init_conv', 'add1', style='dashed', arrowhead='none')
        g.edge('res_blocks', 'post_res_conv')
        g.edge('post_res_conv', 'add1')
        g.edge('add1', 'upsample1')
        g.edge('upsample1', 'upsample2')
        g.edge('upsample2', 'final_conv')
        g.edge('final_conv', 'sr_output')

    # --- Discriminator Subgraph ---
    with dot.subgraph(name='cluster_discriminator') as d:
        d.attr(label='Discriminator', style='filled', color='lightgrey')
        d.node('hr_input', 'High-Res Image', **styles['input'])
        d.node('conv_blocks', '8 x Conv Blocks\n(Downsampling)', **styles['conv'])
        d.node('dense_layers', 'Dense Layers', **styles['conv'])
        d.node('d_output', 'Real/Fake Score\n(Sigmoid)', **styles['output'])
        
        d.edge('hr_input', 'conv_blocks')
        d.edge('conv_blocks', 'dense_layers')
        d.edge('dense_layers', 'd_output')

    # Connect Generator to Discriminator
    dot.edge('sr_output', 'conv_blocks', style='dashed')
    
    # Render the graph
    dot.render('model_1_srgan', format='png', view=False, cleanup=True)
    print("Generated model_1_srgan.png")


def create_esrgan_diagram():
    """Generates the diagram for Model 2: ESRGAN-style."""
    dot = graphviz.Digraph(
        'ESRGAN_Model',
        comment='Model 2: ESRGAN-style Architecture',
        graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.8'}
    )

    # --- Generator Subgraph ---
    with dot.subgraph(name='cluster_generator') as g:
        g.attr(label='Generator (Model 2 - ESRGAN)', style='filled', color='lightgrey')
        
        # Input
        g.node('lr_input', 'Low-Res Image', **styles['input'])

        # Main Path
        g.node('init_conv', 'Initial Conv', **styles['conv'])
        g.node('rrdb_blocks', '23 x RRDB Blocks\n(NO Batch Norm)', **styles['special_block'])
        g.node('post_rrdb_conv', 'Post-RRDB Conv', **styles['conv'])
        g.node('add1', '+', **styles['op'])
        g.node('upsample1', 'Upsampling (PixelShuffle)', **styles['special_block'])
        g.node('upsample2', 'Upsampling (PixelShuffle)', **styles['special_block'])
        g.node('final_convs', 'Refinement Convs', **styles['conv'])
        g.node('sr_output', 'Super-Resolved Image', **styles['output'])

        # RRDB detail (as a separate cluster for clarity)
        with g.subgraph(name='cluster_rrdb') as rrdb:
            rrdb.attr(label='Residual-in-Residual Dense Block (RRDB)', style='filled', color='#f0f0f0')
            rrdb.node('drb1', 'Dense Block', **styles['special_block'])
            rrdb.node('drb2', 'Dense Block', **styles['special_block'])
            rrdb.node('drb3', 'Dense Block', **styles['special_block'])
            rrdb.node('rrdb_add', '+ (beta=0.2)', **styles['op'])
            rrdb.edge('rrdb_blocks', 'drb1', style='dashed', label='(example)')
            rrdb.edge('drb1', 'drb2')
            rrdb.edge('drb2', 'drb3')
            rrdb.edge('drb3', 'rrdb_add')

        # Connect generator parts
        g.edge('lr_input', 'init_conv')
        g.edge('init_conv', 'rrdb_blocks')
        g.edge('init_conv', 'add1', style='dashed', arrowhead='none')
        g.edge('rrdb_blocks', 'post_rrdb_conv')
        g.edge('post_rrdb_conv', 'add1')
        g.edge('add1', 'upsample1')
        g.edge('upsample1', 'upsample2')
        g.edge('upsample2', 'final_convs')
        g.edge('final_convs', 'sr_output')

    # --- Discriminator Subgraph ---
    with dot.subgraph(name='cluster_discriminator') as d:
        d.attr(label='Discriminator (RaGAN)', style='filled', color='lightgrey')
        d.node('hr_input', 'High-Res Image', **styles['input'])
        d.node('conv_blocks', '8 x Conv Blocks\n(Downsampling)', **styles['conv'])
        d.node('dense_layers', 'Dense Layers', **styles['conv'])
        d.node('d_output', 'Realism Logit\n(No Sigmoid)', **styles['output'])
        
        d.node('loss_label', 'Loss compares Realism Logits\nof real vs. fake batches', **styles['label'])
        
        d.edge('hr_input', 'conv_blocks')
        d.edge('conv_blocks', 'dense_layers')
        d.edge('dense_layers', 'd_output')

    # Connect Generator to Discriminator
    dot.edge('sr_output', 'conv_blocks', style='dashed')
    
    # Render the graph
    dot.render('model_2_esrgan', format='png', view=False, cleanup=True)
    print("Generated model_2_esrgan.png")


if __name__ == '__main__':
    create_srgan_diagram()
    create_esrgan_diagram()