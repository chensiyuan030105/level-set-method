import taichi as ti
from zalesak_simulator import *
from singlevortex_simulator import *
from ferrofluid_simulator import *
from apic_extension import *
from initializer_2d import *
from visualizer_2d import *

if __name__ == '__main__':
    res = 100
    # solver = FerrofluidSimulator(2, (res, res), 1e-4, 1, 3.1250e-3)
    # solver = ZalesakSimulator(2, (res, res), 1e-4, 1, 3.1250e-3)
    solver = SingleVortexSimulator(2, (res, res), 1e-4, 1, 1 / (res - 1))
    # solver = APICSimulator(2, (res, res), 1e-4, 1, 1 / res)
    # initializer = SphereInitializer2D(res, 0.5, 0.5, 0.2) 
    # x0, y0, x1, y1 = 0.08, 0.2, 0.92, 0.45
    # initializer = Initializer2D(res, x0 + 0.03, y0 + 0.035, x1 - 0.02, y1 - 0.02, \
    #     [[x0, y0, x1, y0 + 0.02], \
    #      [x0, y0 + 0.02, x0 + 0.02, y1 + 0.02],
    #      [x1 - 0.02, y0 + 0.02, x1, y1 + 0.02]])
    
    initializer = InitializerSingleVortex2D(res, 0.5, 0.75, 0.15)

    visualizer = GUIVisualizer2D(res, 512, 'levelset', True)
    solver.initialize(initializer)
    solver.run(10000, visualizer)
    # ti.kernel_profiler_print()