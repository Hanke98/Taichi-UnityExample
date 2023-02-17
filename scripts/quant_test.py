import taichi as ti
import argparse
import math
import struct

parser = argparse.ArgumentParser()
parser.add_argument("--aot", action='store_true', default=False)
parser.add_argument("-q", "--quant", action='store_true', default=False)
args = parser.parse_args()

aot = args.aot
quant = args.quant

arch = ti.vulkan
ti.init(arch=arch)

dim = 2
n_particles = 10000

x_arr = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))

def get_sca(_x):
    return (_x.get_scalar_field(i) for i in range(dim))


if quant:
    # qfxt = ti.types.quant.fixed(bits=32, max_value=2)
    # x = ti.field(dtype=qfxt)

    # bitpack = ti.BitpackedFields(max_num_bits=32)
    # bitpack.place(x)
    # # ti.root.place(bitpack)
    # ti.root.dense(ti.i, n_particles).place(bitpack)

    qfxt = ti.types.quant.fixed(bits=16, max_value=1.2)

    x = ti.Matrix.field(dim, dim, dtype=qfxt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(x.get_scalar_field(0, 0), x.get_scalar_field(0, 1))
    bitpack2 = ti.BitpackedFields(max_num_bits=32)
    bitpack2.place(x.get_scalar_field(1, 0), x.get_scalar_field(1, 1))
    ti.root.dense(ti.i, n_particles).place(bitpack, bitpack2)

else:
    # x = ti.field(ti.f32, shape=(n_particles))
    x = ti.Matrix.field(dim, dim, ti.f32, shape=(n_particles))


@ti.kernel
def get_pos(x_arr: ti.types.ndarray(field_dim=1)):
    # x_arr[0][0] = x[None]
    for i in range(n_particles):
        x_arr[i][0] = x[i][0, 0]
        # x_arr[i][1] = x[i][0, 1] 


@ti.kernel
def test1():
    # x[None] = 0.7
    # x[None] = x[None] + 0.4

    # for i in range(n_particles):
    #     x[i] = 0.7
    #     x[i] = x[i] + 0.5

    x_ = 1.0
    # x[0][0, 0] = 1.0
    # for i in range(n_particles):
    #     x[i][0, 0] = 1.0
    #     x[i][0, 1] = 0.0
        # x[i][1, 0] = 0.0
        # x[i][1, 1] = 1.0

@ti.kernel
def rotate_18_degrees():
    angle = math.pi / 2 
    for i in range(n_particles):
        x_new = ti.Matrix([[0.8, 0.4], [0.0, 1.0]])
        # x_new = x[i] @ ti.Matrix(
        #     [[ti.cos(angle), ti.sin(angle)], 
        #      [-ti.sin(angle), ti.cos(angle)]])
        # for j in ti.static(range(dim)):
            # for k in ti.static(range(dim)):
            #     x[i][j, k] = x_new[j, k]
        x[i][0, 0] = 0.1
        x[i][0, 1] = 0.4

def compile_aot():
    global arch
    sym_x_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                            'x_arr',
                            ti.f32,
                            field_dim=1,
                            element_shape=(3, ))

    g_init_builder = ti.graph.GraphBuilder()
    init_step = g_init_builder.create_sequential()
    init_step.dispatch(test1)
    init_step.dispatch(rotate_18_degrees)
    g_init_builder.append(init_step)
    
    g_init = g_init_builder.compile()

    g_update_builder = ti.graph.GraphBuilder()
    g_update_builder.dispatch(get_pos, sym_x_arr)
    g_update = g_update_builder.compile()

    mod = ti.aot.Module(arch)
    mod.add_graph('init', g_init)
    mod.add_graph('update', g_update)
    return mod, g_init, g_update

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def run(g_init, g_update):
    g_init.run({})
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        g_update.run({"x_arr": x_arr})
        x_arr_np = x_arr.to_numpy()[:, :2]
        print(x_arr_np[0, 0], float_to_hex(x_arr_np[0, 0]))

        gui.circles(x_arr_np,
                    radius=1.5,
                    color=0x068587)
        gui.show()
    

if __name__ == "__main__":
    mod, g_init, g_update = compile_aot()
    if aot:
        mod.archive(f"Assets/Resources/TaichiModules/quant_test_quant_{quant}.cgraph.tcm")
    else:
        run(g_init, g_update)



