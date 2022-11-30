import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan, print_ir=True)

n = 320
# canvas = ti.ndarray(dtype=ti.f32, shape=(n * 2, n))
# qfxt = ti.types.quant.fixed(bits=8, max_value=2)
# canvas = ti.field(dtype=qfxt)
# bitpack = ti.BitpackedFields(max_num_bits=32)
# bitpack.place(canvas)
# ti.root.dense(ti.ij, (n*2, n)).place(bitpack)

real = ti.float32
canvas = ti.field(dtype=real)
ti.root.dense(ti.ij, (n*2, n)).place(canvas)


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def fractal(t: ti.f32, canvas: ti.types.ndarray(field_dim=2)):
    for i, j in canvas:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        canvas[i, j] = 1 - iterations * 0.02

# sym_t = ti.graph.Arg(ti.graph.ArgKind.SCALAR,
#                      "t",
#                      ti.f32,
#                      element_shape=())
# sym_canvas = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
#                           "canvas",
#                           ti.f32,
#                           field_dim=2,
#                           element_shape=())

# gb = ti.graph.GraphBuilder()
# gb.dispatch(fractal, sym_t, sym_canvas)
# graph = gb.compile()

# mod = ti.aot.Module(ti.vulkan)
# mod.add_graph('fractal', graph)
# mod.archive("Assets/Resources/TaichiModules/fractal.cgraph.tcm")

# gui = ti.GUI("Julia Set", res=(n * 2, n))

# i = 0
# while True:
#     i += 1
#     args = {
#         "t": 0.03 * i,
#         "canvas": canvas,
#     }
#     canvas2 = np.repeat(canvas.to_numpy().reshape(n * 2,n,1), 3, 2)
#     print(canvas2.min())
#     graph.run(args)
#     gui.set_image(canvas2)
#     gui.show()
