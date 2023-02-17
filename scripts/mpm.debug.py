import argparse
import math

import numpy as np
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--aot", action="store_true", default=False)
parser.add_argument("-q", "--quant", action="store_true", default=False)
args = parser.parse_args()

quant = args.quant
aot = args.aot
arch = ti.cpu
# arch = ti.vulkan
ti.init(arch=arch, default_fp=ti.float32)  # , vk_api_version="1.0")

n_particles = 4096
nn = int(math.sqrt(n_particles))
n_grid = 128
dt = 3e-4
dx = 1/n_grid

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho

gravity = 9.8 * 2
bound = 3
dim = 2
E = 200
N_SUBSTEPS = 40
S = 2**30
# SV = 2**30 / 7

x_arr = ti.Vector.ndarray(3, float, shape=(n_particles))


def get_sca(_x):
    return (_x.get_scalar_field(i) for i in range(dim))


def get_sca_mat(_x, i):
    return (_x.get_scalar_field(i, j) for j in range(dim))


if quant:
    qfxt = ti.types.quant.fixed(bits=16, max_value=1.01)
    qfxt2 = ti.types.quant.fixed(bits=16, max_value=7)
    qfxt3 = ti.types.quant.fixed(bits=16, max_value=1e3)
    x = ti.Vector.field(dim, qfxt)

    bitpack = ti.BitpackedFields(max_num_bits=32)
    bitpack.place(*get_sca(x))
    ti.root.dense(ti.i, n_particles).place(bitpack)

    v = ti.Vector.field(dim, qfxt2)
    bitpack_v = ti.BitpackedFields(max_num_bits=32)
    bitpack_v.place(*get_sca(v))
    ti.root.dense(ti.i, n_particles).place(bitpack_v)

    C = ti.Matrix.field(dim, dim, qfxt3)
    bitpack_C0 = ti.BitpackedFields(max_num_bits=32)
    bitpack_C1 = ti.BitpackedFields(max_num_bits=32)
    bitpack_C0.place(*get_sca_mat(C, 0))
    bitpack_C1.place(*get_sca_mat(C, 1))
    ti.root.dense(ti.i, n_particles).place(bitpack_C0, bitpack_C1)

    J = ti.field(ti.f32)
    ti.root.dense(ti.i, n_particles).place(J)
    # ti.root.dense(ti.i, n_particles).place(bitpack, bitpack_v,
    # bitpack_C0, bitpack_C1, J)
else:
    x = ti.Vector.field(dim, float, shape=(n_particles))
    v = ti.Vector.field(dim, float, shape=(n_particles))
    C = ti.Matrix.field(dim, dim, float, shape=(n_particles))
    J = ti.field(float, shape=(n_particles))

grid_v = ti.Vector.field(dim, float, shape=(n_grid, n_grid))
grid_m = ti.field(float, shape=(n_grid, n_grid))

grid_v_int = ti.Vector.field(
    dim, dtype=ti.int32, shape=(n_grid, n_grid)
)  # grid node momentum/velocity
grid_m_int = ti.field(dtype=ti.int32, shape=(n_grid, n_grid))  # grid node mass


@ti.kernel
def get_pos(x_arr: ti.types.ndarray(field_dim=1)):
    for i in range(n_particles):
        for k in ti.static(range(dim)):
            x_arr[i][k] = x[i][k]


@ti.kernel
def init_particles():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1

    # n = nn
    # w = 0.4
    # h = 0.4
    # dw = w / n
    # dh = w / n
    # for i in range(n_particles):
    #     r, c = i / n, i % n
    #     x[i][0] = dw * r + 20 / 128
    #     x[i][1] = dh * c + 20 / 128
    #     v[i] = [0, -0.2]
    #     J[i] = 1


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in ti.ndrange(n_grid, n_grid):
        # if grid_m[i, j] > 0:
        #     grid_v[i, j] /= grid_m[i, j]
        if grid_m[i, j] > 1e-8:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    for p in range(n_particles):
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C
        # for k in ti.static(range(dim)):
        #     v[p][k] = new_v[k]
        # for k in ti.static(range(dim)):
        #     x[p][k] = x[p][k] + dt * v[p][k]


def compile_aot():
    global arch
    sym_x_arr = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "x_arr", ti.f32,
        field_dim=1, element_shape=(3,)
    )

    g_init_builder = ti.graph.GraphBuilder()
    g_init_builder.dispatch(init_particles)
    g_init = g_init_builder.compile()

    g_update_builder = ti.graph.GraphBuilder()
    substep = g_update_builder.create_sequential()


    for _ in range(N_SUBSTEPS):
        g_update_builder.append(substep)

    g_update_builder.dispatch(get_pos, sym_x_arr)
    g_update = g_update_builder.compile()

    mod = ti.aot.Module(arch)
    mod.add_graph("init", g_init)
    mod.add_graph("update", g_update)
    return mod, g_init, g_update


def run(g_init, g_update):
    g_init.run({})
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        g_update.run({"x_arr": x_arr})
        x_arr_np = x_arr.to_numpy()[:, :2]
        # m_ = grid_m.to_numpy(dtype=np.float32)
        # v_ = grid_v.to_numpy(dtype=np.float32)
        # print(f'max_m: { m_.max()}')
        # print(f'max_v: {v_.max(axis=0)}')
        # print(f'min_v: {v_.min(axis=0)}')
        gui.circles(x_arr_np, radius=1.5, color=0x068587)
        gui.show()


def run_gui():
    init_particles()
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        x_np = x.to_numpy()
        for _ in range(N_SUBSTEPS):
            substep()

        ti.sync()
        gui.circles(x_np, radius=1.5, color=0x068587)
        gui.show()


if __name__ == "__main__":
    run_gui()
