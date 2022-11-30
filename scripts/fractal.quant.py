import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan, print_ir=True)

real = ti.float32
n = 320 * 4
dim = 3
_cvs = ti.ndarray(dtype=real, shape=(n * 2, n))
qfxt = ti.types.quant.fixed(bits=8, max_value=2)
canvas = ti.Vector.field(dim, dtype=qfxt)
bitpack = ti.BitpackedFields(max_num_bits=32)
bitpack.place(canvas)
ti.root.dense(ti.ij, (n*2, n)).place(bitpack)


# canvas = ti.Vector.field(dim, dtype=real)
# ti.root.dense(ti.ij, (n*2, n)).place(canvas)

t = ti.field(float, shape=())


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def fractal():
    t[None] += 0.03
    for i, j in canvas:
    # for i, j in canvas:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t[None]) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        for k in ti.static(range(dim)):
            canvas[i, j][k] = 1 - iterations * 0.02

@ti.kernel
def get_data(cvs: ti.types.ndarray(field_dim=2)):
    for i, j in canvas:
        cvs[i, j] = canvas[i, j][0]


# sym_t = ti.graph.Arg(ti.graph.ArgKind.SCALAR,
#                      "t",
#                      ti.f32,
#                      element_shape=())
sym_canvas = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                          "cvs",
                          ti.f32,
                          field_dim=2,
                          element_shape=())

gb = ti.graph.GraphBuilder()
gb.dispatch(fractal)
# gb.dispatch(fractal, sym_t, sym_canvas)
graph = gb.compile()

get_graph_builder = ti.graph.GraphBuilder()
get_graph_builder.dispatch(get_data, sym_canvas)
get_graph = get_graph_builder.compile()

mod = ti.aot.Module(ti.vulkan)
mod.add_graph('fractal', graph)
mod.add_graph('get_data', get_graph)
mod.archive("Assets/Resources/TaichiModules/fractal.cgraph.quant.tcm")

gui = ti.GUI("Julia Set", res=(n * 2, n))

i = 0
while True:
    i += 1
    args = {
        "t": 0.03 * i,
        "canvas": canvas,
    }
    # canvas2 = np.repeat(canvas.to_numpy(dtype=np.float32).reshape(n * 2,n,1), 3, 2)
    # canvas2 = canvas.to_numpy(dtype=np.float32)
    # t = 0.03 * i
    fractal()
    get_data(_cvs)

    canvas2 = np.repeat(_cvs.to_numpy().reshape(n * 2,n,1), 3, 2)

    # graph.run(args)
    gui.set_image(canvas2)
    gui.show()
