import taichi as ti
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--aot", action='store_true', default=False)
args = parser.parse_args()

aot = args.aot
# arch = ti.cpu
arch = ti.vulkan
ti.init(arch=arch)

n_particles = 8192
n_grid = 64
dt = 2e-4

p_rho = 1
gravity = 9.8
bound = 3
dim = 2
E = 400
N_SUBSTEPS = 50

x_arr = ti.Vector.ndarray(3, ti.f32, shape=(n_particles))

x = ti.Vector.field(2, ti.f32, shape=(n_particles))
v = ti.Vector.field(2, ti.f32, shape=(n_particles))
C = ti.Matrix.field(2, 2, ti.f32, shape=(n_particles))
J = ti.field(ti.f32, shape=(n_particles))

grid_v = ti.Vector.field(2, ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(ti.f32, shape=(n_grid, n_grid))


@ti.kernel
def get_pos(x_arr: ti.types.ndarray(field_dim=1)):
    for i in range(n_particles):
        for k in ti.static(range(dim)):
            x_arr[i][k] = x[i][k]
        x_arr[i][2] = 0


@ti.kernel
def init_particles():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1

@ti.kernel
def substep_reset_grid():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0


@ti.kernel
def substep_p2g():
    for p in x:
        dx = 1 / grid_v.shape[0]
        p_vol = (dx * 0.5)**2
        p_mass = p_vol * p_rho
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
            grid_v[base +
                   offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass


@ti.kernel
def substep_update_grid_v():
    for i, j in grid_m:
        num_grid = grid_v.shape[0]
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > num_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > num_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0


@ti.kernel
def substep_g2p():
    for p in x:
        dx = 1 / grid_v.shape[0]
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


def compile_aot():
    global arch
    sym_x_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                            'x_arr',
                            ti.f32,
                            field_dim=1,
                            element_shape=(3, ))

    g_init_builder = ti.graph.GraphBuilder()
    g_init_builder.dispatch(init_particles)
    g_init = g_init_builder.compile()

    g_update_builder = ti.graph.GraphBuilder()
    substep = g_update_builder.create_sequential()

    substep.dispatch(substep_reset_grid)
    substep.dispatch(substep_p2g)
    substep.dispatch(substep_update_grid_v)
    substep.dispatch(substep_g2p)

    for i in range(N_SUBSTEPS):
        g_update_builder.append(substep)

    g_update_builder.dispatch(get_pos, sym_x_arr)
    g_update = g_update_builder.compile()

    mod = ti.aot.Module(arch)
    mod.add_graph('init', g_init)
    mod.add_graph('update', g_update)
    return mod, g_init, g_update

def run(g_init, g_update):
    g_init.run({})
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        g_update.run({"x_arr": x_arr})
        x_arr_np = x_arr.to_numpy()[:, :2]
        gui.circles(x_arr_np,
                    radius=1.5,
                    color=0x068587)
        gui.show()
    

if __name__ == "__main__":
    mod, g_init, g_update = compile_aot()
    if aot:
        mod.archive("Assets/Resources/TaichiModules/mpm88.cgraph.tcm")
    else:
        run(g_init, g_update)

