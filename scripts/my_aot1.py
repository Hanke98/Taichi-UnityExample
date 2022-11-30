import taichi as ti

# ti.init(arch=ti.cpu)
ti.init(arch=ti.vulkan, vk_api_version="1.0", print_ir=True)
dim = 2

real = ti.float32
N = 20000000
x = ti.Vector.ndarray(dim, real, shape=N)
y = ti.Vector.ndarray(dim, real, shape=N)
r = ti.Vector.ndarray(dim, real, shape=N)

# N = 20000000
# quant 45 mu
# float32 45 mu 
# qfxt = ti.types.quant.fixed(bits=8, max_value=2)
# z = ti.Vector.field(4, qfxt)

# bitpack = ti.BitpackedFields(max_num_bits=32)
# bitpack.place(z.get_scalar_field(0), z.get_scalar_field(1), 
#               z.get_scalar_field(2), z.get_scalar_field(3))

# ti.root.dense(ti.i, N).place(bitpack)
z = ti.Vector.field(4, real, shape=N)

@ti.kernel
def set_value():
    for i in range(N):
        for k in ti.static(range(4)):
            z[i][k] = 1.0


@ti.kernel
def init(_x: ti.types.ndarray(field_dim=1), _y: ti.types.ndarray(field_dim=1)):
    for i in range(N):
        for k in ti.static(range(dim)):
            _x[i][k] = 0.5#z[i]
            _y[i][k] = -0.5


@ti.kernel
def func(_x: ti.types.ndarray(field_dim=1), _y: ti.types.ndarray(field_dim=1), _r: ti.types.ndarray(field_dim=1)):
    for i in range(N):
        for k in ti.static(range(dim)):
            _r[i][k] = _x[i][k] + _y[i][k]


sym_x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'x', real, field_dim=1, element_shape=(dim,))
sym_y = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'y', real, field_dim=1, element_shape=(dim,))
sym_r = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'r', real, field_dim=1, element_shape=(dim,))


g_init_builder = ti.graph.GraphBuilder()
# g_init_builder.dispatch(set_value)
g_init_builder.dispatch(init, sym_x, sym_y)
g_init_builder.dispatch(func, sym_x, sym_y, sym_r)

g_init = g_init_builder.compile()
# g_init.run({'x': x, 'y': y, 'r': r})


assigning_builder = ti.graph.GraphBuilder()
assigning_builder.dispatch(set_value)
assgining = assigning_builder.compile()

mod = ti.aot.Module(ti.vulkan)
mod.add_graph('init', g_init)
mod.add_graph('assign', assgining)
mod.archive("Assets/Resources/TaichiModules/my_aot.cgraph.tcm")

